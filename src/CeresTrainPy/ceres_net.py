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
from typing import Tuple, NamedTuple

import torch
import torch.nn as nn
from torch import nn

import lightning as pl
from lightning.fabric import Fabric

from activation_functions import Swish, ReLUSquared
from losses import LossCalculator
from encoder_layer import EncoderLayer
from config import Configuration
from mlp2_layer import MLP2Layer
from rms_norm import RMSNorm

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
  def __init__(self, n_alphas):
    super().__init__()
    self.n_alphas = n_alphas
    alphas = torch.zeros((n_alphas,))
    alphas[-1] = 1.0
    self.alphas = torch.nn.Parameter(alphas)

  def forward(self, all_previous_x):
    weighted_avg = all_previous_x[0] * self.alphas[0]
    for i in range(1, self.n_alphas):
      weighted_avg += self.alphas[i] * all_previous_x[i]
    return weighted_avg


class Head(nn.Module):
    def __init__(self, Activation, IN_SIZE, FC_SIZE, OUT_SIZE):
        super(Head, self).__init__()

        self.fc = nn.Linear(IN_SIZE, FC_SIZE)
        self.fcActivation = Activation
        self.fcFinal = nn.Linear(FC_SIZE, OUT_SIZE)

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
    q_ratio
):
    """
    CeresNet is a transformer architecture network module for chess built with PyTorch Lightning. 
    """
    super().__init__()

    self.fabric = fabric
    self.save_hyperparameters()

    self.MAX_MOVES = 1858
    self.NUM_INPUT_BYTES_PER_SQUARE = 137 # N.B. also update in train.py

    self.NUM_TOKENS_INPUT = 64
    self.NUM_TOKENS_NET = 64   
  
    self.DROPOUT_RATE = config.Exec_DropoutRate
    self.EMBEDDING_DIM = config.NetDef_ModelDim
    self.NUM_LAYERS = config.NetDef_NumLayers


    self.TRANSFORMER_OUT_DIM = self.EMBEDDING_DIM * self.NUM_TOKENS_NET

    self.NUM_HEADS = config.NetDef_NumHeads
    self.FFN_MULT = config.NetDef_FFNMultiplier
    self.DEEPNORM = config.NetDef_DeepNorm
    self.denseformer = config.NetDef_DenseFormer
    self.prior_state_dim = config.NetDef_PriorStateDim
    self.moves_left_loss_weight = moves_left_loss_weight
    self.q_deviation_loss_weight = q_deviation_loss_weight
    self.uncertainty_policy_weight = uncertainty_policy_weight
    self.action_uncertainty_loss_weight = action_uncertainty_loss_weight

    
    if (config.NetDef_HeadsActivationType == 'ReLU'):
      self.Activation = torch.nn.ReLU()
    elif (config.NetDef_HeadsActivationType == 'ReLUSquared'):
      self.Activation = ReLUSquared()
    elif (config.NetDef_HeadsActivationType == 'Swish'):
      self.Activation = Swish()
    elif (config.NetDef_HeadsActivationType == 'Mish'):
      self.Activation = torch.nn.Mish()
    else:
      raise Exception('Unknown activation type', config.NetDef_HeadsActivationType)
    self.test = config.Exec_TestFlag

    self.embedding_layer = nn.Linear(self.NUM_INPUT_BYTES_PER_SQUARE + self.prior_state_dim, self.EMBEDDING_DIM)
    self.embedding_layer2 = None if self.NUM_TOKENS_NET == self.NUM_TOKENS_INPUT else nn.Linear(self.NUM_INPUT_BYTES_PER_SQUARE, self.EMBEDDING_DIM)
    self.embedding_norm = torch.nn.LayerNorm(self.EMBEDDING_DIM, eps=1E-6) if config.NetDef_NormType == 'LayerNorm' else RMSNorm(self.EMBEDDING_DIM, eps=1E-6)

    HEAD_MULT = config.NetDef_HeadWidthMultiplier

    HEAD_PREMAP_DIVISOR = 64
    self.HEAD_PREMAP_PER_SQUARE = (HEAD_MULT * self.EMBEDDING_DIM) // HEAD_PREMAP_DIVISOR
    self.headPremap = nn.Linear(self.EMBEDDING_DIM, self.HEAD_PREMAP_PER_SQUARE)

    HEAD_SHARED_LINEAR_DIV = 4
    self.HEAD_IN_SIZE = 64 * (self.HEAD_PREMAP_PER_SQUARE // HEAD_SHARED_LINEAR_DIV)
    self.headSharedLinear = nn.Linear(64 * self.HEAD_PREMAP_PER_SQUARE, self.HEAD_IN_SIZE)

    self.policy_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858)
    
    if self.prior_state_dim > 0:
      self.state_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 64*self.prior_state_dim)

    if action_loss_weight > 0:
      self.action_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858 * 3)

    if action_uncertainty_loss_weight > 0:
      self.action_uncertainty_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858)

    self.value_head = Head(self.Activation, self.HEAD_IN_SIZE, 64 * HEAD_MULT, 3)
    self.value2_head = Head(self.Activation, self.HEAD_IN_SIZE, 64 * HEAD_MULT, 3)     
    self.unc_head = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1)

    if self.uncertainty_policy_weight > 0:
      self.unc_policy = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1)
    
    if moves_left_loss_weight > 0:
      self.mlh_head = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1)

    if q_deviation_loss_weight > 0:      
      self.qdev_upper = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1)
      self.qdev_lower = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1)



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
      self.smolgenPrepLayer = nn.Linear(SMOLGEN_INTERMEDIATE_DIM // config.NetDef_SmolgenToHeadDivisor, self.NUM_TOKENS_NET * self.NUM_TOKENS_NET)
    else:
      self.smolgenPrepLayer = None

    num_tokens_q = self.NUM_TOKENS_NET
    num_tokens_kv = self.NUM_TOKENS_NET
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
      self.dwa_modules = torch.nn.ModuleList([DWA(n_alphas=i+2) for i in range(self.NUM_LAYERS)])
 

  def forward(self, squares: torch.Tensor, prior_state:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(squares, list):
      # when saving/restoring from ONNX the input will appear as a list instead of sequence of arguments
#      squares = squares[0]
#      prior_state = squares[1]
      squares = squares[0]
      
    flow = squares

#    if self.test:
#      flow[:, :, 109] = 0
#      flow[:, :, 110] = 0
#      flow[:, :, 111] = 0

#    flow[:, :, 119] = 0 # QNegativeBlunders
#    flow[:, :, 120] = 0 # QPositiveBlunders

#      condition = (flow[:, :, 109] <0.01) & (flow[:, :, 110] < 0.3) & (flow[:, :, 111] < 0.3)
#      flow[:, :, 107] = condition.bfloat16()
#      flow[:, :, 108] = 1 - flow[:, :, 107]
      

    # Embedding layer.
    flow_squares = flow.reshape(-1, self.NUM_TOKENS_INPUT, (self.NUM_TOKENS_INPUT * self.NUM_INPUT_BYTES_PER_SQUARE) // self.NUM_TOKENS_INPUT)

    if self.prior_state_dim > 0:
      # Append prior state to the input if is available for this position.
      append_tensor = prior_state if prior_state is not None else torch.zeros(squares.shape[0], self.NUM_TOKENS_INPUT, self.prior_state_dim).to(flow.device).to(torch.bfloat16)
      append_tensor = append_tensor.reshape(squares.shape[0], self.NUM_TOKENS_INPUT, self.prior_state_dim)
      flow_squares = torch.cat((flow_squares, append_tensor), dim=-1)

    flow = self.embedding_layer(flow_squares)

    if self.NUM_TOKENS_NET > self.NUM_TOKENS_INPUT:
      flow2 = self.embedding_layer2(flow_squares)
      flow = torch.cat([flow, flow2], 1)
      
    flow = self.embedding_norm(flow)
      
    if self.denseformer:
      all_previous_x = [flow]
      
    # Main transformer body (stack of encoder layers)
    for i in range(self.NUM_LAYERS):
      # Main policy encoder block
      flow, _ = self.transformer_layer[i](flow, None)

      if self.denseformer:
        all_previous_x.append(flow)
        flow = self.dwa_modules[i](all_previous_x)
      
  
    # Heads.
    flattenedSquares = self.headPremap(flow)
    flattenedSquares = flattenedSquares.reshape(-1, 64 * self.HEAD_PREMAP_PER_SQUARE)
    flattenedSquares = self.headSharedLinear(flattenedSquares)
    
    policy_out = self.policy_head(flattenedSquares)
    value_out = self.value_head(flattenedSquares)
    value2_out = self.value2_head(flattenedSquares)
    unc_out = self.unc_head(flattenedSquares)
    unc_policy_out = self.unc_policy(flattenedSquares) if self.uncertainty_policy_weight > 0 else unc_out # unc_out is just a dummy so not None

    # Note that if these heads are not used we use a fill-in tensor (borrowed from unc) 
    # to avoid None values that might be problematic in export (especially ONNX)
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
    
    
    # Possibly create a blended value target for Value2.
    # The intention is to slightly soften the noisy and hard wdl_nondeblundered target.
    #wdl_blend = (wdl_nondeblundered * 0.70 + wdl_deblundered * 0.15 + wdl_q * 0.15)
    wdl_blend = wdl_nondeblundered
    
    value_target = wdl_q * self.q_ratio + wdl_deblundered * (1 - self.q_ratio)
    p_loss = 0 if policy_out is None else loss_calc.policy_loss(policy_target, policy_out, SUBTRACT_ENTROPY)
    v_loss = 0 if value_out is None else loss_calc.value_loss(value_target, value_out, SUBTRACT_ENTROPY)
    v2_loss = 0 if value2_out is None else loss_calc.value2_loss(wdl_blend, value2_out, SUBTRACT_ENTROPY)
    ml_loss = 0 if moves_left_out is None else loss_calc.moves_left_loss(moves_left_target, moves_left_out)
    u_loss = 0 if unc_out is None else loss_calc.unc_loss(unc_target, unc_out)
    q_deviation_lower_loss = 0 if q_deviation_lower_out is None else loss_calc.q_deviation_lower_loss(q_deviation_lower_target, q_deviation_lower_out)
    q_deviation_upper_loss = 0 if q_deviation_upper_out is None else loss_calc.q_deviation_upper_loss(q_deviation_upper_target, q_deviation_upper_out)

    # We have two value scores and want them to be consistent modulo inversion (prior_board and this_board).
    # The value of this board is taken to be "more definitive" so it is the target (however this assumes policy was correct....)
    value_diff_loss = 0 if self.value_diff_loss_weight == 0 or prior_value_out == None else loss_calc.value_diff_loss(value_out, prior_value_out, SUBTRACT_ENTROPY)
    value2_diff_loss = 0 if self.value2_diff_loss_weight == 0 or prior_value2_out == None else loss_calc.value2_diff_loss(value2_out, prior_value2_out, SUBTRACT_ENTROPY)

    if action_target is not None:
      action_loss = 0 if self.action_loss_weight == 0 else multiplier_action_loss * loss_calc.action_loss(action_target, action_out, SUBTRACT_ENTROPY)
      action_uncertainty_loss = 0 if self.action_uncertainty_loss_weight == 0 else self.action_uncertainty_loss_weight * loss_calc.action_unc_loss(torch.abs(action_target - action_out), action_uncertainty_out)
    else:
      action_loss = 0
      action_uncertainty_loss = 0
      
    uncertainty_policy_loss = 0 if self.uncertainty_policy_weight == 0 else loss_calc.uncertainty_policy_loss(uncertainty_policy_target, uncertainty_policy_out)

    total_loss = (self.policy_loss_weight * p_loss
        + self.value_loss_weight * v_loss
        + self.moves_left_loss_weight * ml_loss
        + self.unc_loss_weight * u_loss
        + self.value2_loss_weight * v2_loss
        + self.q_deviation_loss_weight * q_deviation_lower_loss
        + self.q_deviation_loss_weight * q_deviation_upper_loss
        + self.value_diff_loss_weight * value_diff_loss
        + self.value2_diff_loss_weight * value2_diff_loss
        + self.action_loss_weight * action_loss
        + self.action_uncertainty_loss_weight * action_uncertainty_loss
        + self.uncertainty_policy_weight * uncertainty_policy_loss
        )
        
    if (log_stats):
        policy_accuracy = 0 if policy_out is None else loss_calc.calc_accuracy(policy_target, policy_out, True)
        value_accuracy = 0 if value_out is None else loss_calc.calc_accuracy(value_target, value_out, False)
        self.fabric.log("pos_mm", num_pos // 1000000., step=num_pos)
        self.fabric.log("LR", last_lr[0], step=num_pos)
        self.fabric.log("total_loss", total_loss, step=num_pos)
        self.fabric.log("policy_loss", p_loss,  step=num_pos)
        self.fabric.log("policy_acc",policy_accuracy,  step=num_pos)
        self.fabric.log("value_acc",value_accuracy,  step=num_pos)
        self.fabric.log("value_loss", v_loss,  step=num_pos)
        self.fabric.log("value2_loss", v2_loss,  step=num_pos)
        self.fabric.log("moves_left_loss", ml_loss, step=num_pos)
        self.fabric.log("unc_loss", u_loss, step=num_pos)
        self.fabric.log("unc_policy_loss", uncertainty_policy_loss, step=num_pos)
        self.fabric.log("q_deviation_lower_loss", q_deviation_lower_loss, step=num_pos)
        self.fabric.log("q_deviation_upper_loss", q_deviation_upper_loss, step=num_pos)
        self.fabric.log("value_diff_loss", value_diff_loss, step=num_pos)
        self.fabric.log("value2_diff_loss", value2_diff_loss, step=num_pos)
        self.fabric.log("action_loss", action_loss, step=num_pos)
        self.fabric.log("action_uncertainty_loss", action_uncertainty_loss, step=num_pos)
        
    return total_loss
                    