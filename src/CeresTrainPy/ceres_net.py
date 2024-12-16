# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

# NOTE: this module is derived from: https://github.com/Rocketknight1/minimal_lczero.

import math
from multiprocessing import Value
from typing import Tuple, NamedTuple

import torch
import torch.nn as nn
from torch import nn

import lightning as pl
from lightning.fabric import Fabric
from lightning.pytorch.utilities import grad_norm

from activation_functions import Swish, ReLUSquared
from losses import LossCalculator
from encoder_layer import EncoderLayer
from config import Configuration
from mlp2_layer import MLP2Layer
from rms_norm import RMSNorm
from lora import LoRALinear

from config import NUM_TOKENS_INPUT, NUM_TOKENS_NET, NUM_INPUT_BYTES_PER_SQUARE


"""
Code from:
  "DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging," Pagliardini et. al.  
    https://arxiv.org/pdf/2402.02622v2.pdf
This code is taken directly from the paper, not from their github repository.

The github code would be advisable or necessary for very deep nets (50 to 100 layers)
to improve performance and reduce memory usage.

However this is unnecessary for Ceres nets which are not as deep.
Also the more complex github implementation uses in place operations that are not supported by torch.compile.
"""
class DWA(torch.nn.Module): 
  def __init__(self, n_alphas, depth=None):
    super().__init__()
    self.n_alphas = n_alphas
    alphas = torch.zeros((n_alphas,))
    alphas[-1] = 1.0
    if depth is not None:
      alphas = alphas.unsqueeze(1)
      alphas = alphas.repeat(1, depth)
    self.alphas = torch.nn.Parameter(alphas)

  def forward(self, all_previous_x):
    weighted_avg = all_previous_x[0] * self.alphas[0]
    for i in range(1, self.n_alphas):
      weighted_avg += self.alphas[i] * all_previous_x[i]
    return weighted_avg


class Head(nn.Module):
  def __init__(self, Activation, IN_SIZE, FC_SIZE, OUT_SIZE, lora_rank_divisor):
    super(Head, self).__init__()

    self.fc = nn.Linear(IN_SIZE, FC_SIZE)
    if lora_rank_divisor > 0:
      self.fc = LoRALinear(self.fc, lora_rank_divisor, True)

    self.fcActivation = Activation

    self.fcFinal = nn.Linear(FC_SIZE, OUT_SIZE)
    if lora_rank_divisor > 0:
      self.fcFinal = LoRALinear(self.fcFinal, lora_rank_divisor, True)

  def forward(self, flow):
    flow = self.fc(flow)
    flow = self.fcActivation(flow)
    flow = self.fcFinal(flow)
    return flow


class CeresNet(pl.LightningModule):
  def __init__(
    self,
    fabric : Fabric,
    config : Configuration,
    policy_loss_weight,
    value_loss_weight,
    moves_left_loss_weight,
    unc_loss_weight,
    value2_loss_weight,
    q_deviation_loss_weight,
    value_diff_loss_weight,
    value2_diff_loss_weight,
    action_loss_weight,
    uncertainty_policy_weight,
    action_uncertainty_loss_weight,
    q_ratio):
    """
    CeresNet is a transformer architecture network module for chess built with PyTorch Lightning. 
    """
    super().__init__()

    self.fabric = fabric
    self.save_hyperparameters()
    self.config = config
     
    self.DROPOUT_RATE = config.Exec_DropoutRate
    self.EMBEDDING_DIM = config.NetDef_ModelDim
    self.NUM_LAYERS = config.NetDef_NumLayers


    self.TRANSFORMER_OUT_DIM = self.EMBEDDING_DIM * NUM_TOKENS_NET

    self.NUM_HEADS = config.NetDef_NumHeads
    self.FFN_MULT = config.NetDef_FFNMultiplier
    self.DEEPNORM = config.NetDef_DeepNorm
    self.denseformer = config.NetDef_DenseFormer
    self.prior_state_dim = config.NetDef_PriorStateDim
    self.moves_left_loss_weight = moves_left_loss_weight
    self.q_deviation_loss_weight = q_deviation_loss_weight
    self.value2_loss_weight = value2_loss_weight
    self.uncertainty_policy_weight = uncertainty_policy_weight
    self.action_uncertainty_loss_weight = action_uncertainty_loss_weight

    
    if config.NetDef_HeadsActivationType == 'ReLU':
      self.Activation = torch.nn.ReLU()
    elif config.NetDef_HeadsActivationType == 'ReLUSquared':
      self.Activation = ReLUSquared()
    elif config.NetDef_HeadsActivationType == 'Swish':
      self.Activation = Swish()
    elif config.NetDef_HeadsActivationType == 'Mish':
      self.Activation = torch.nn.Mish()
    elif config.NetDef_HeadsActivationType == 'Identity':
      self.Activation = torch.nn.Identity()
    else:
      raise Exception('Unknown activation type', config.NetDef_HeadsActivationType)
    self.test = config.Exec_TestFlag

    self.embedding_layer = nn.Linear(NUM_INPUT_BYTES_PER_SQUARE + self.prior_state_dim, self.EMBEDDING_DIM)
    self.embedding_layer2 = None if NUM_TOKENS_NET == NUM_TOKENS_INPUT else nn.Linear(NUM_INPUT_BYTES_PER_SQUARE, self.EMBEDDING_DIM)
    self.embedding_norm = torch.nn.LayerNorm(self.EMBEDDING_DIM, eps=1E-6) if config.NetDef_NormType == 'LayerNorm' else RMSNorm(self.EMBEDDING_DIM, eps=1E-6)

    HEAD_MULT = config.NetDef_HeadWidthMultiplier

    HEAD_PREMAP_DIVISOR = 64
    self.HEAD_PREMAP_PER_SQUARE = (HEAD_MULT * self.EMBEDDING_DIM) // HEAD_PREMAP_DIVISOR
    self.headPremap = nn.Linear(self.EMBEDDING_DIM, self.HEAD_PREMAP_PER_SQUARE)

    HEAD_SHARED_LINEAR_DIV = 4
    self.HEAD_IN_SIZE = 64 * (self.HEAD_PREMAP_PER_SQUARE // HEAD_SHARED_LINEAR_DIV)
    self.headSharedLinear = nn.Linear(64 * self.HEAD_PREMAP_PER_SQUARE, self.HEAD_IN_SIZE)

    self.policy_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858, config.Opt_LoRARankDivisor)
    
    if self.prior_state_dim > 0:
      self.state_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 64*self.prior_state_dim, config.Opt_LoRARankDivisor)

    if action_loss_weight > 0:
      self.action_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858 * 3, config.Opt_LoRARankDivisor)

    if action_uncertainty_loss_weight > 0:
      self.action_uncertainty_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858, config.Opt_LoRARankDivisor)

    self.value_head = Head(self.Activation, self.HEAD_IN_SIZE, 64 * HEAD_MULT, 3, config.Opt_LoRARankDivisor)
    self.unc_head = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1, config.Opt_LoRARankDivisor)

    if self.value2_loss_weight > 0:
      self.value2_head = Head(self.Activation, 2 + self.HEAD_IN_SIZE, 64 * HEAD_MULT, 3, config.Opt_LoRARankDivisor) 

    if self.uncertainty_policy_weight > 0:
      self.unc_policy = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1, config.Opt_LoRARankDivisor)
    
    if moves_left_loss_weight > 0:
      self.mlh_head = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1, config.Opt_LoRARankDivisor)

    if q_deviation_loss_weight > 0:      
      self.qdev_upper = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1, config.Opt_LoRARankDivisor)
      self.qdev_lower = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1, config.Opt_LoRARankDivisor)



    if self.DEEPNORM:     
      self.alpha = math.pow(2 * self.NUM_LAYERS, 0.25)
    else:      
      self.alpha = 1

    SMOLGEN_PER_SQUARE_DIM = config.NetDef_SmolgenDimPerSquare
    SMOLGEN_INTERMEDIATE_DIM = config.NetDef_SmolgenDim

    ATTENTION_MULTIPLIER = config.NetDef_AttentionMultiplier

    if config.NetDef_SoftMoE_NumExperts > 0:
      assert config.NetDef_SoftMoE_MoEMode in ("None", "ReplaceLinear", "AddLinearSecondLayer", "ReplaceLinearSecondLayer"), 'implementation restriction: only AddLinearSecondLayer currently supported'
      assert config.NetDef_SoftMoE_NumSlotsPerExpert == 1
      assert config.NetDef_SoftMoE_UseBias == True
      assert config.NetDef_SoftMoE_UseNormalization == False
      assert config.NetDef_SoftMoE_OnlyForAlternatingLayers == True
      
    EPS = 1E-6
    
    if SMOLGEN_PER_SQUARE_DIM > 0 and SMOLGEN_INTERMEDIATE_DIM > 0:
      self.smolgenPrepLayer = nn.Linear(SMOLGEN_INTERMEDIATE_DIM // config.NetDef_SmolgenToHeadDivisor, NUM_TOKENS_NET * NUM_TOKENS_NET)
    else:
      self.smolgenPrepLayer = None

    if config.NetDef_UseRPE or config.NetDef_UseRelBias:
      self.rpe_factor_shared = torch.nn.Parameter(make_rpe_map(), requires_grad=False)
    else:
      self.rpe_factor_shared = None

    num_tokens_q = NUM_TOKENS_NET
    num_tokens_kv = NUM_TOKENS_NET
    self.transformer_layer = torch.nn.Sequential(
       *[EncoderLayer('T', num_tokens_q, num_tokens_kv,
                      self.NUM_LAYERS, self.EMBEDDING_DIM,
                      self.FFN_MULT*self.EMBEDDING_DIM, 
                      self.NUM_HEADS,
                      ffn_activation_type = config.NetDef_FFNActivationType, 
                      norm_type = config.NetDef_NormType, layernorm_eps=EPS, 
                      attention_multiplier = ATTENTION_MULTIPLIER,
                      smoe_mode = config.NetDef_SoftMoE_MoEMode,
                      smoe_num_experts = config.NetDef_SoftMoE_NumExperts,
                      smolgen_per_square_dim = SMOLGEN_PER_SQUARE_DIM, 
                      smolgen_intermediate_dim = SMOLGEN_INTERMEDIATE_DIM, 
                      smolgen_head_divisor = config.NetDef_SmolgenToHeadDivisor,
                      smolgenPrepLayer = self.smolgenPrepLayer, 
                      smolgen_activation_type = config.NetDef_SmolgenActivationType,
                      alpha=self.alpha, layerNum=i, dropout_rate=self.DROPOUT_RATE,
                      use_rpe=config.NetDef_UseRPE, 
                      use_rpe_v=config.NetDef_UseRPE_V,
                      rpe_factor_shared=self.rpe_factor_shared,
                      use_rel_bias=config.NetDef_UseRelBias,
                      use_nonlinear_attention=config.NetDef_NonLinearAttention,
                      dual_attention_mode = config.NetDef_DualAttentionMode if not config.Exec_TestFlag else (config.NetDef_DualAttentionMode if i % 2 == 1 else 'None'),
                      test = config.Exec_TestFlag)
        for i in range(self.NUM_LAYERS)])

    self.policy_loss_weight = policy_loss_weight
    self.value_loss_weight = value_loss_weight
    self.moves_left_loss_weight = moves_left_loss_weight
    self.unc_loss_weight = unc_loss_weight
    self.value2_loss_weight = value2_loss_weight
    self.q_deviation_loss_weight = q_deviation_loss_weight
    self.value_diff_loss_weight = value_diff_loss_weight
    self.value2_diff_loss_weight = value2_diff_loss_weight
    self.action_loss_weight = action_loss_weight
    self.uncertainty_policy_weight = uncertainty_policy_weight
    self.action_uncertainty_loss_weight = action_uncertainty_loss_weight
    self.q_ratio = q_ratio

    if (self.denseformer):
      self.dwa_modules = torch.nn.ModuleList([DWA(n_alphas=i+2, depth=self.EMBEDDING_DIM) for i in range(self.NUM_LAYERS)])
 

  def forward(self, squares: torch.Tensor, prior_state:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(squares, list):
      # when saving/restoring from ONNX the input will appear as a list instead of sequence of arguments
#      squares = squares[0]
#      prior_state = squares[1]
      squares = squares[0]
      
    flow = squares
    qblunders_negative_positive = squares[:, 0, 119:121].clone().view(-1, 2)

    # non-inplace version of the next two lines 
    # required to avoid problems with ONNX execution
    flow = torch.cat((flow[:, :, :119], torch.zeros_like(flow[:, :, 119:121]), flow[:, :, 121:]), dim=2)
#    flow[:, :, 119] = 0 # QNegativeBlunders
#    flow[:, :, 120] = 0 # QPositiveBlunders

#    condition = (flow[:, :, 109] <0.01) & (flow[:, :, 110] < 0.3) & (flow[:, :, 111] < 0.3)
#    flow[:, :, 107] = condition.bfloat16()
#    flow[:, :, 108] = 1 - flow[:, :, 107]
      

    # Embedding layer.
    flow_squares = flow.reshape(-1, NUM_TOKENS_INPUT, (NUM_TOKENS_INPUT * NUM_INPUT_BYTES_PER_SQUARE) // NUM_TOKENS_INPUT)

    if self.prior_state_dim > 0:
      # Append prior state to the input if is available for this position.
      append_tensor = prior_state if prior_state is not None else torch.zeros(squares.shape[0], NUM_TOKENS_INPUT, self.prior_state_dim).to(flow.device).to(torch.bfloat16)
      append_tensor = append_tensor.reshape(squares.shape[0], NUM_TOKENS_INPUT, self.prior_state_dim)
      flow_squares = torch.cat((flow_squares, append_tensor), dim=-1)

    flow = self.embedding_layer(flow_squares)

    if NUM_TOKENS_NET > NUM_TOKENS_INPUT:
      flow2 = self.embedding_layer2(flow_squares)
      flow = torch.cat([flow, flow2], 1)
      
    flow = self.embedding_norm(flow)
      
    if self.denseformer:
      all_previous_x = [flow]
      
    # Main transformer body (stack of encoder layers)
    for i in range(self.NUM_LAYERS):
      flow = self.transformer_layer[i](flow)

      if self.denseformer:
        all_previous_x.append(flow)
        flow = self.dwa_modules[i](all_previous_x)
      
  
    # Heads.
    flattenedSquares = self.headPremap(flow)
    flattenedSquares = flattenedSquares.reshape(-1, 64 * self.HEAD_PREMAP_PER_SQUARE)
    flattenedSquares = self.headSharedLinear(flattenedSquares)
    
    # Note that if these heads are not used we use a fill-in tensor (borrowed from unc or value) 
    # to avoid None values that might be problematic in export (especially ONNX)
    policy_out = self.policy_head(flattenedSquares)
    value_out = self.value_head(flattenedSquares)
    value2_out = self.value2_head(torch.cat((flattenedSquares, qblunders_negative_positive), -1)) if self.value2_loss_weight > 0 else value_out
    unc_out = self.unc_head(flattenedSquares)
    unc_policy_out = self.unc_policy(flattenedSquares) if self.uncertainty_policy_weight > 0 else unc_out # unc_out is just a dummy so not None

    action_out             = self.action_head(flattenedSquares).reshape(-1, 1858, 3) if self.action_loss_weight > 0 else unc_out
    action_uncertainty_out = self.action_uncertainty_head(flattenedSquares) if self.action_uncertainty_loss_weight > 0 else unc_out
    state_out              = self.state_head(flattenedSquares) if self.prior_state_dim > 0 else unc_out
    moves_left_out         = self.mlh_head(flattenedSquares) if self.moves_left_loss_weight > 0 else unc_out
    q_deviation_lower_out = self.qdev_lower(flattenedSquares) if self.q_deviation_loss_weight > 0 else unc_out
    q_deviation_upper_out = self.qdev_upper(flattenedSquares) if self.q_deviation_loss_weight > 0 else unc_out   

    ret = policy_out, value_out, moves_left_out, unc_out, value2_out, q_deviation_lower_out, q_deviation_upper_out, unc_policy_out, action_out, state_out, action_uncertainty_out

    return ret


  def compute_loss(self, loss_calc : LossCalculator, batch, policy_out, value_out, moves_left_out, unc_out,
                    value2_out, q_deviation_lower_out, q_deviation_upper_out, uncertainty_policy_out,
                    prior_value_out, prior_value2_out,
                    action_target, action_out, action_uncertainty_out,
                    multiplier_action_loss,
                    num_pos, last_lr, log_stats):

    # if we are logging statistics, make two passes, the first of which 
    # calculates and logs individual gradient norms for each loss
    LOG_PER_LOSS_GRADIENT_NORMS = False # N.B. this feature only works with non-compiled models run on a single GPU
                                        #      and slows down training significantly, so only use for quick tests 
    if LOG_PER_LOSS_GRADIENT_NORMS and log_stats and self.fabric.is_global_zero:
      self.compute_loss_or_gradnorm(loss_calc, batch, policy_out, value_out, moves_left_out, unc_out,
                                    value2_out, q_deviation_lower_out, q_deviation_upper_out, uncertainty_policy_out,
                                    prior_value_out, prior_value2_out,
                                    action_target, action_out, action_uncertainty_out,
                                    multiplier_action_loss,
                                    num_pos, last_lr, log_stats, gradient_norm_logging_mode = True)
       
    return self.compute_loss_or_gradnorm (loss_calc, batch, policy_out, value_out, moves_left_out, unc_out,
                                          value2_out, q_deviation_lower_out, q_deviation_upper_out, uncertainty_policy_out,
                                          prior_value_out, prior_value2_out,
                                          action_target, action_out, action_uncertainty_out,
                                          multiplier_action_loss,
                                          num_pos, last_lr, log_stats, gradient_norm_logging_mode = False)


  def compute_loss_or_gradnorm(self, loss_calc : LossCalculator, batch, policy_out, value_out, moves_left_out, unc_out,
                               value2_out, q_deviation_lower_out, q_deviation_upper_out, uncertainty_policy_out,
                               prior_value_out, prior_value2_out,
                               action_target, action_out, action_uncertainty_out,
                               multiplier_action_loss,
                               num_pos, last_lr, log_stats, gradient_norm_logging_mode):
    policy_target = batch['policies']
    wdl_deblundered = batch['wdl_deblundered']
    wdl_q = batch['wdl_q']
    moves_left_target = batch['mlh']
    unc_target = batch['unc']
    wdl_nondeblundered = batch['wdl_nondeblundered']
    uncertainty_policy_target = batch['uncertainty_policy']
    q_deviation_lower_target = batch['q_deviation_lower']
    q_deviation_upper_target = batch['q_deviation_upper']
    
    #	Subtract entropy from cross entropy to insulate loss magnitude 
    #	from distributional shift and make the loss more interpretable 
    #	because it takes out the portion that is irreducible.
    SUBTRACT_ENTROPY = True

   
    # Note that the loss weights are passed into the loss calculation functions in loss_calc module.
    # But they are only used for informational purposes and NOT applied to the losses applied by these functions.
    # Instead, the loss weights are only applied in the weighted average calculation in the assignment to total_loss.
    # Therefore the values logged (e.g. to Tensorboard) are the raw (unweighted) losses 
    # which are invariant to the particular weights in use (to facilitate comparison across different training runs).

    # Possibly create a blended value target for Value2.
    # The intention is to slightly soften the noisy and hard wdl_nondeblundered target.
    wdl_blend = (wdl_nondeblundered * 0.70 + wdl_deblundered * 0.15 + wdl_q * 0.15)
    #wdl_blend = wdl_nondeblundered  
    value_target = wdl_q * self.q_ratio + wdl_deblundered * (1 - self.q_ratio)

    p_loss = 0 if policy_out is None else loss_calc.policy_loss(policy_target, policy_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.policy_loss_weight)
    v_loss = 0 if value_out is None else loss_calc.value_loss(value_target, value_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value_loss_weight)
    v2_loss = 0 if value2_out is None else loss_calc.value2_loss(wdl_blend, value2_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value2_loss_weight)
    ml_loss = 0 if moves_left_out is None else loss_calc.moves_left_loss(moves_left_target, moves_left_out, gradient_norm_logging_mode, self.moves_left_loss_weight)
    u_loss = 0 if unc_out is None else loss_calc.unc_loss(unc_target, unc_out, gradient_norm_logging_mode, self.unc_loss_weight)
    q_deviation_lower_loss = 0 if q_deviation_lower_out is None else loss_calc.q_deviation_lower_loss(q_deviation_lower_target, q_deviation_lower_out, gradient_norm_logging_mode, self.q_deviation_loss_weight)
    q_deviation_upper_loss = 0 if q_deviation_upper_out is None else loss_calc.q_deviation_upper_loss(q_deviation_upper_target, q_deviation_upper_out, gradient_norm_logging_mode, self.q_deviation_loss_weight)


    if self.config.NetDef_TrainOn4BoardSequences:
      # TO DO: probably the multiplier_action_loss should somehow be propagated into the gradient norms when these are calculated
      action_loss = multiplier_action_loss * loss_calc.action_loss(action_target, action_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.action_loss_weight)
      action_uncertainty_loss = multiplier_action_loss * self.action_uncertainty_loss_weight * loss_calc.action_unc_loss(torch.abs(action_target - action_out), action_uncertainty_out, gradient_norm_logging_mode, self.action_uncertainty_loss_weight)
      # We have two value scores and want them to be consistent modulo inversion (prior_board and this_board).
      # The value of this board is taken to be "more definitive" so it is the target (however this assumes policy was correct....)
      value_diff_loss = 0 if self.value_diff_loss_weight == 0 or prior_value_out == None else loss_calc.value_diff_loss(value_out, prior_value_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value_diff_loss_weight)
      value2_diff_loss = 0 if self.value2_diff_loss_weight == 0 or prior_value2_out == None else loss_calc.value2_diff_loss(value2_out, prior_value2_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value2_diff_loss_weight)
    else:
      action_loss = 0
      action_uncertainty_loss = 0
      value_diff_loss = 0
      value2_diff_loss = 0

    uncertainty_policy_loss = 0 if uncertainty_policy_out is None else loss_calc.uncertainty_policy_loss(uncertainty_policy_target, uncertainty_policy_out, gradient_norm_logging_mode, self.uncertainty_policy_weight)

    total_loss = (self.policy_loss_weight * p_loss
        + self.value_loss_weight * v_loss
        + self.value2_loss_weight * v2_loss
        + self.moves_left_loss_weight * ml_loss
        + self.unc_loss_weight * u_loss
        + self.q_deviation_loss_weight * q_deviation_lower_loss
        + self.q_deviation_loss_weight * q_deviation_upper_loss
        + self.value_diff_loss_weight * value_diff_loss
        + self.value2_diff_loss_weight * value2_diff_loss
        + self.action_loss_weight * action_loss
        + self.action_uncertainty_loss_weight * action_uncertainty_loss
        + self.uncertainty_policy_weight * uncertainty_policy_loss)
        
    if (log_stats):
      if not gradient_norm_logging_mode:
        stat_suffix = ""
        policy_accuracy = 0 if policy_out is None else loss_calc.calc_accuracy(policy_target, policy_out, True)
        value_accuracy = 0 if value_out is None else loss_calc.calc_accuracy(value_target, value_out, False)
        self.fabric.log("pos_mm", num_pos // 1000000., step=num_pos)
        self.fabric.log("LR", last_lr, step=num_pos)
        self.fabric.log("total_loss", total_loss, step=num_pos)

        # Log GPU (CUDA) statistics
        if torch.cuda.is_available():
          for gpu_num in range(torch.cuda.device_count()):
            # Note: we enumerate all devices on the host instead of only those used 
            #       for this training run (self.config.Exec_DeviceIDs) for two reasons:
            #       1. The torch.cuda numbering scheme (highest performance at lowest index)
            #          is potentially different from the what the application sees
            #          (unless the enviroment variable is set: CUDA_DEVICE_ORDER=PCI_BUS_ID).
            #       2. Potentially (for power and thermal reasons) it is useful to monitor all devices
            #          even if not used by this training run.                   
            try:
              self.fabric.log("gpu_temp_"+str(gpu_num), torch.cuda.temperature(gpu_num), step=num_pos)
              self.fabric.log("gpu_power_draw_"+str(gpu_num), torch.cuda.power_draw(gpu_num)/1000, step=num_pos)
              self.fabric.log("gpu_utilization_"+str(gpu_num), torch.cuda.utilization(gpu_num), step=num_pos)
              #self.fabric.log("gpu_memory_used_"+str(gpu_num), torch.cuda.memory_usage(gpu_num), step=num_pos)
              #self.fabric.log("gpu_clock_rate_"+str(gpu_num), torch.cuda.clock_rate(gpu_num), step=num_pos)
            except:
              pass # requires pynvml, may fail e.g. on Windows    
      else:
        stat_suffix = "_gnorm"

      if not gradient_norm_logging_mode:
        self.fabric.log("policy_acc" + stat_suffix,policy_accuracy,  step=num_pos)
        self.fabric.log("value_acc" + stat_suffix,value_accuracy,  step=num_pos)

      self.fabric.log("policy_loss" + stat_suffix, p_loss,  step=num_pos)
      self.fabric.log("value_loss" + stat_suffix, v_loss,  step=num_pos)
      self.fabric.log("value2_loss" + stat_suffix, v2_loss,  step=num_pos)
      self.fabric.log("moves_left_loss" + stat_suffix, ml_loss, step=num_pos)
      self.fabric.log("unc_loss" + stat_suffix, u_loss, step=num_pos)
      self.fabric.log("unc_policy_loss" + stat_suffix, uncertainty_policy_loss, step=num_pos)
      self.fabric.log("q_deviation_lower_loss" + stat_suffix, q_deviation_lower_loss, step=num_pos)
      self.fabric.log("q_deviation_upper_loss" + stat_suffix, q_deviation_upper_loss, step=num_pos)
      self.fabric.log("value_diff_loss" + stat_suffix, value_diff_loss, step=num_pos)
      self.fabric.log("value2_diff_loss" + stat_suffix, value2_diff_loss, step=num_pos)
      self.fabric.log("action_loss" + stat_suffix, action_loss, step=num_pos)
      self.fabric.log("action_uncertainty_loss" + stat_suffix, action_uncertainty_loss, step=num_pos)

    return total_loss


  
"""
Prepare static relative position (RPE) encoding map.
This RPE idea and initialization code taken from work of Daniel Monroe, see:
https://github.com/Ergodice/lczero-training/blob/a7271f25a1bd84e5e22bf924f7365cd003cb8d2f/tf/tfprocess.py
""" 
def make_rpe_map():
  # 15 * 15 in units for distance pairs to 64 * 64 pairs of squares
  # (rounded from 15 up to 16 to be a power of 2)
  out = torch.zeros((16*16, 64*64))
  for i in range(8):
    for j in range(8):
      for k in range(8):
        for l in range(8):
          out[15 * (i - k + 7) + (j - l + 7), 64 * (i * 8 + j) + k * 8 + l] = 1
  return out


