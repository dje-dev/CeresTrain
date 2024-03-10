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
    q_ratio,
    optimizer,
    learning_rate
):
    """
    CeresNet is a transformer architecture network module for chess built with PyTorch Lightning. 

    Parameters:
        fabric (Fabric): Underlying lightning fabric instance
        config (Configuration): Configuration settings for the network (read from JSON)
        policy_loss_weight: Weight for the policy loss component.
        value_loss_weight: Weight for the value loss component.
        moves_left_loss_weight: Weight for the moves left loss component.
        unc_loss_weight: Weight for the uncertainty loss component.
        value2_loss_weight: Weight for the secondary value loss component.
        q_deviation_loss_weight: Weight for the Q deviation (lower and upper) loss components.
        q_ratio: Fraction of weight to be put on Q (search value results) versus game WDL result.
        optimizer: Optimizer to be used for training.
        learning_rate: Learning rate for the optimizer.
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
    
    if (config.NetDef_HeadsActivationType == 'ReLU'):
      self.Activation = torch.nn.ReLU()
    elif (config.NetDef_HeadsActivationType == 'ReLUSquared'):
      self.Activation = ReLUSquared()
    elif (config.NetDef_HeadsActivationType == 'Swish'):
      self.Activation = Swish()
    else:
      raise Exception('Unknown activation type', config.NetDef_HeadsActivationType)

    self.embedding_layer = nn.Linear(self.NUM_INPUT_BYTES_PER_SQUARE, self.EMBEDDING_DIM)
    self.embedding_layer2  = None if self.NUM_TOKENS_NET == self.NUM_TOKENS_INPUT else nn.Linear(self.NUM_INPUT_BYTES_PER_SQUARE, self.EMBEDDING_DIM)
    self.global_dim = config.NetDef_GlobalStreamDim
    self.heads_pure_global = config.NetDef_GlobalStreamDim > 0 and config.NetDef_HeadsNonPolicyGlobalStreamOnly
    self.embedding_layer_global = None if config.NetDef_GlobalStreamDim == 0 else nn.Linear(self.NUM_TOKENS_INPUT * self.NUM_INPUT_BYTES_PER_SQUARE, config.NetDef_GlobalStreamDim) 
    
    HEAD_MULT = config.NetDef_HeadWidthMultiplier

    self.HEAD_PREMAP_DIVISOR_POLICY = 8
    FINAL_POLICY_FC1_SIZE = 128 * HEAD_MULT
    FINAL_POLICY_FC2_SIZE = 64 * HEAD_MULT
    self.policyHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY)

    self.fcPolicyFinal1 = nn.Linear(config.NetDef_GlobalStreamDim + self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY, FINAL_POLICY_FC1_SIZE)
    self.fcPolicyRELU1 = self.Activation
    self.fcPolicyFinal2 = nn.Linear(FINAL_POLICY_FC1_SIZE, FINAL_POLICY_FC2_SIZE)
    self.fcPolicyRELU2 = self.Activation
    self.fcPolicyFinal3 = nn.Linear(FINAL_POLICY_FC2_SIZE, 1858)

    self.HEAD_PREMAP_DIVISOR_VALUE = 8
    FINAL_VALUE_FC1_SIZE = 32 * HEAD_MULT
    FINAL_VALUE_FC2_SIZE = 8 * HEAD_MULT

    GLOBAL_ONLY = config.NetDef_GlobalStreamDim > 0 and config.NetDef_HeadsNonPolicyGlobalStreamOnly
    
    if GLOBAL_ONLY:
      self.out_value_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_VALUE_FC1_SIZE)
    else:
      self.valueHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY)
      self.out_value_layer1 = nn.Linear(config.NetDef_GlobalStreamDim + self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_VALUE,FINAL_VALUE_FC1_SIZE)     
    self.relu_value_1 = self.Activation
    self.out_value_layer2 = nn.Linear(FINAL_VALUE_FC1_SIZE, FINAL_VALUE_FC2_SIZE)
    self.relu_value_2 = self.Activation 
    self.out_value_layer3 = nn.Linear(FINAL_VALUE_FC2_SIZE, 3)

    if GLOBAL_ONLY:
      self.out_value2_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_VALUE_FC1_SIZE)      
    else:
      self.value2HeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY)
      self.out_value2_layer1 = nn.Linear(config.NetDef_GlobalStreamDim + self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_VALUE,FINAL_VALUE_FC1_SIZE)
    self.relu_value2_1 = self.Activation
    self.out_value2_layer2 = nn.Linear(FINAL_VALUE_FC1_SIZE, FINAL_VALUE_FC2_SIZE)
    self.relu_value2_2 = self.Activation 
    self.out_value2_layer3 = nn.Linear(FINAL_VALUE_FC2_SIZE,3)

    self.HEAD_PREMAP_DIVISOR_MLH = 16
    FINAL_MLH_FC1_SIZE = 32 * HEAD_MULT
    FINAL_MLH_FC2_SIZE = 8 * HEAD_MULT

    if GLOBAL_ONLY:
      self.out_mlh_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_MLH_FC1_SIZE)
    else:  
      self.mlhHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_MLH)
      self.out_mlh_layer1 = nn.Linear(config.NetDef_GlobalStreamDim + self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_MLH,FINAL_MLH_FC1_SIZE)
    self.relu_mlh = self.Activation
    self.out_mlh_layer2 = nn.Linear(FINAL_MLH_FC1_SIZE, FINAL_MLH_FC2_SIZE)
    self.out_mlh_layer3 = nn.Linear(FINAL_MLH_FC2_SIZE, 1)

    self.HEAD_PREMAP_DIVISOR_UNC = 16
    FINAL_UNC_FC1_SIZE = 32 * HEAD_MULT
    FINAL_UNC_FC2_SIZE = 8 * HEAD_MULT

    if GLOBAL_ONLY:
      self.out_unc_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_UNC_FC1_SIZE)
    else:
      self.uncHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_UNC)
      self.out_unc_layer1 = nn.Linear(config.NetDef_GlobalStreamDim + self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_UNC,FINAL_UNC_FC1_SIZE)
    self.relu_unc = self.Activation
    self.out_unc_layer2 = nn.Linear(FINAL_UNC_FC1_SIZE, FINAL_UNC_FC2_SIZE)
    self.out_unc_layer3 = nn.Linear(FINAL_UNC_FC2_SIZE, 1)

    self.HEAD_PREMAP_DIVISOR_QDEV = 16
    FINAL_QDEV_FC1_SIZE = 32 * HEAD_MULT
    FINAL_QDEV_FC2_SIZE = 8 * HEAD_MULT

    if GLOBAL_ONLY:
      self.out_qdev_lower_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_QDEV_FC1_SIZE)        
    else:
      self.qDevLowerHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_QDEV)
      self.out_qdev_lower_layer1 = nn.Linear(config.NetDef_GlobalStreamDim + self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_QDEV, FINAL_QDEV_FC1_SIZE)
    self.relu_qdev_lower = self.Activation
    self.out_qdev_lower_layer2 = nn.Linear(FINAL_QDEV_FC1_SIZE,FINAL_QDEV_FC2_SIZE)
    self.out_qdev_lower_layer3 = nn.Linear(FINAL_QDEV_FC2_SIZE, 1)

    if GLOBAL_ONLY:
      self.out_qdev_upper_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_QDEV_FC1_SIZE)      
    else: 
      self.qDevUpperHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_QDEV)
      self.out_qdev_upper_layer1 = nn.Linear(config.NetDef_GlobalStreamDim + self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_QDEV, FINAL_QDEV_FC1_SIZE)
    self.relu_qdev_upper = self.Activation
    self.out_qdev_upper_layer2 = nn.Linear(FINAL_QDEV_FC1_SIZE, FINAL_QDEV_FC2_SIZE)
    self.out_qdev_upper_layer3 = nn.Linear(FINAL_QDEV_FC2_SIZE, 1)

    self.DEEPNORM = config.NetDef_DeepNorm

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

    if config.NetDef_UseRPE:
      RPE_INNER_DIM = 512
      self.rpe_factor_q = nn.Linear(RPE_INNER_DIM, 64*64, bias=False)
      self.rpe_factor_k = self.rpe_factor_q # shared
      self.rpe_factor_v = None
    else:
      self.smolgenPrepLayer = None

    num_tokens_q = self.NUM_TOKENS_NET
    num_tokens_kv = self.NUM_TOKENS_NET
    self.transformer_layer = torch.nn.Sequential(
       *[EncoderLayer('T', num_tokens_q, num_tokens_kv,
                      self.NUM_LAYERS, self.EMBEDDING_DIM, config.NetDef_GlobalStreamDim,
                      self.FFN_MULT*self.EMBEDDING_DIM, 
                      self.NUM_HEADS * 2 if config.Exec_TestFlag and i % 2 == 0 else self.NUM_HEADS,
                      ffn_activation_type = config.NetDef_FFNActivationType, 
                      norm_type = config.NetDef_NormType, layernorm_eps=EPS, 
                      attention_multiplier = ATTENTION_MULTIPLIER * 2 if config.Exec_TestFlag and i % 2 == 0 else ATTENTION_MULTIPLIER,
                      global_stream_attention_per_square = config.NetDef_GlobalStreamAttentionPerSquare,
                      smoe_mode = config.NetDef_SoftMoE_MoEMode,
                      smoe_num_experts = config.NetDef_SoftMoE_NumExperts,
                      smolgen_per_square_dim = SMOLGEN_PER_SQUARE_DIM, 
                      smolgen_intermediate_dim = SMOLGEN_INTERMEDIATE_DIM, 
                      smolgen_head_divisor = config.NetDef_SmolgenToHeadDivisor,
                      smolgenPrepLayer = self.smolgenPrepLayer, 
                      smolgen_activation_type = config.NetDef_SmolgenActivationType,
                      alpha=self.alpha, layerNum=i, dropout_rate=self.DROPOUT_RATE,
                      use_rpe=config.NetDef_UseRPE, 
                      rpe_factor_q = self.rpe_factor_q if config.NetDef_UseRPE else None,
                      rpe_factor_k = self.rpe_factor_k if config.NetDef_UseRPE else None,
                      rpe_factor_v = self.rpe_factor_v if config.NetDef_UseRPE else None,
                      dual_attention_mode = config.NetDef_DualAttentionMode if config.Exec_TestFlag and i % 2 == 1 else 'None',
                      test = config.Exec_TestFlag)
        for i in range(self.NUM_LAYERS)])

    if config.NetDef_GlobalStreamDim > 0:
      PER_SQUARE_DIM = 16
      NUM_GLOBAL_INNER = config.NetDef_GlobalStreamFFNMultiplier * config.NetDef_GlobalStreamDim
      GLOBAL_FFN_ACTIVATION_TYPE = 'ReLU' # Note that squared RelU may cause training instabilities (?)
      self.mlp_global = torch.nn.Sequential(*[MLP2Layer(model_dim=config.NetDef_GlobalStreamDim + self.NUM_TOKENS_NET * PER_SQUARE_DIM, ffn_inner_dim=NUM_GLOBAL_INNER, out_dim = config.NetDef_GlobalStreamDim, activation_type=GLOBAL_FFN_ACTIVATION_TYPE) for i in range(self.NUM_LAYERS)])
      self.ln_global = torch.nn.LayerNorm(model_dim=config.NetDef_GlobalStreamDim, eps=1e-5) if config.NetDef_NormType == 'LayerNorm' else RMSNorm(config.NetDef_GlobalStreamDim, eps=1e-5)

    self.policy_loss_weight = policy_loss_weight
    self.value_loss_weight = value_loss_weight
    self.moves_left_loss_weight = moves_left_loss_weight
    self.unc_loss_weight = unc_loss_weight
    self.value2_loss_weight = value2_loss_weight
    self.q_deviation_loss_weight = q_deviation_loss_weight
    self.q_ratio = q_ratio
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.test = config.Exec_TestFlag

#    if (self.test):
#      self.pos_encoding = nn.Linear(16, self.EMBEDDING_DIM)
    

  def forward(self, input_planes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(input_planes, list):
      # when saving/restoring from ONNX the input will appear as a list instead of sequence of arguments
      input_planes = input_planes[0]

    flow = input_planes

    # Embedding layer.
    flow_squares = flow.reshape(-1, self.NUM_TOKENS_INPUT, (self.NUM_TOKENS_INPUT * self.NUM_INPUT_BYTES_PER_SQUARE) // self.NUM_TOKENS_INPUT)

    flow = self.embedding_layer(flow_squares)

    if self.NUM_TOKENS_NET > self.NUM_TOKENS_INPUT:
      flow2 = self.embedding_layer2(flow_squares)
      flow = torch.cat([flow, flow2], 1)
      
#    if (self.test):
#      flow_position = flow_squares[:, :, -16:]
#      flow = flow + self.pos_encoding(flow_position)

    if self.global_dim > 0:
      flow_global = input_planes.reshape(-1, self.NUM_TOKENS_INPUT * self.NUM_INPUT_BYTES_PER_SQUARE)
      flow_global = self.embedding_layer_global(flow_global)
    else:
      flow_global = None  
      

    # Main transformer body (stack of encoder layers)
    for i in range(self.NUM_LAYERS):
      # Main policy encoder block, also receive back an update to be applied to the global stream
      flow, global_update = self.transformer_layer[i](flow, flow_global)#.detach())
 
      # Update state based on the output of the encoder layer. 
      if self.global_dim > 0:# &&  NetTransformerLayerEncoder.PER_SQUARE_REDUCED_DIM_TO_GLOBAL_STREAM > 0)  
        # Prepare input to global encoder, concatenating last global state and the update from the policy encoder
        global_update = global_update.reshape(-1,self.NUM_TOKENS_NET * 16)
        flow_state_input = torch.cat([flow_global, global_update], 1)  

        (_, flow_global_new) = self.mlp_global[i](flow_state_input)  
        flow_global = flow_global_new + flow_global
        flow_global = self.ln_global(flow_global)

    # Heads.
    headOut = self.policyHeadPremap(flow)
    flattenedPolicy = headOut.reshape(-1, self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY)
    if self.global_dim > 0:
      flattenedPolicy = torch.concat([flattenedPolicy, flow_global.detach()], 1);       

    ff1Policy = self.fcPolicyFinal1(flattenedPolicy)
    ff1RELUPolicy = self.fcPolicyRELU1(ff1Policy)
    ff2Policy = self.fcPolicyFinal2(ff1RELUPolicy)
    ff2RELUPolicy = self.fcPolicyRELU2(ff2Policy)
    policy_out = self.fcPolicyFinal3(ff2RELUPolicy)

    GLOBAL_ONLY = self.heads_pure_global

     
    if GLOBAL_ONLY:
      flow = flow.detach()
    
    if GLOBAL_ONLY:
      value_out = self.out_value_layer1(flow_global)
    else:      
      value_out = self.valueHeadPremap(flow)
      value_out = value_out.reshape(-1, self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_VALUE)
      if self.global_dim > 0:
        value_out = torch.concat([value_out, flow_global], 1);       
      value_out = self.out_value_layer1(value_out)
    value_out = self.relu_value_1(value_out)
    value_out = self.out_value_layer2(value_out)
    value_out = self.relu_value_2(value_out)
    value_out = self.out_value_layer3(value_out)

    if GLOBAL_ONLY:
      value2_out = self.out_value2_layer1(flow_global)
    else:      
      value2_out = self.value2HeadPremap(flow)
      value2_out = value2_out.reshape(-1, self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_VALUE)
      if self.global_dim > 0:
        value2_out = torch.concat([value2_out, flow_global], 1);       
      value2_out = self.out_value2_layer1(value2_out)
    value2_out = self.relu_value2_1(value2_out)
    value2_out = self.out_value2_layer2(value2_out)
    value2_out = self.relu_value2_2(value2_out)
    value2_out = self.out_value2_layer3(value2_out)

    if GLOBAL_ONLY:
      moves_left_out = self.out_mlh_layer1(flow_global)
    else:      
      moves_left_out = self.mlhHeadPremap(flow)
      moves_left_out = moves_left_out.reshape(-1, self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_MLH)
      if self.global_dim > 0:
        moves_left_out = torch.concat([moves_left_out, flow_global], 1);       
      moves_left_out = self.out_mlh_layer1(moves_left_out)
    moves_left_out = self.relu_mlh(moves_left_out)
    moves_left_out = self.out_mlh_layer2(moves_left_out)
    moves_left_out = self.relu_mlh(moves_left_out)
    moves_left_out = self.out_mlh_layer3(moves_left_out)
    moves_left_out = torch.nn.functional.relu(moves_left_out) # truncate at zero, can't be negative

    if GLOBAL_ONLY:
      unc_out = self.out_unc_layer1(flow_global)
    else:      
      unc_out = self.uncHeadPremap(flow)
      unc_out = unc_out.reshape(-1, self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_UNC)
      if self.global_dim > 0:
        unc_out = torch.concat([unc_out, flow_global], 1);       
      unc_out = self.out_unc_layer1(unc_out)
    unc_out = self.relu_unc(unc_out)
    unc_out = self.out_unc_layer2(unc_out)
    unc_out = self.relu_unc(unc_out)
    unc_out = self.out_unc_layer3(unc_out)
    unc_out = torch.nn.functional.relu(unc_out) # truncate at zero, can't be negative

    if GLOBAL_ONLY:
      q_deviation_lower_out = self.out_qdev_lower_layer1(flow_global)
    else:      
      q_deviation_lower_out = self.qDevLowerHeadPremap(flow)
      q_deviation_lower_out = q_deviation_lower_out.reshape(-1, self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_QDEV)
      if self.global_dim > 0:
        q_deviation_lower_out = torch.concat([q_deviation_lower_out, flow_global], 1);       
      q_deviation_lower_out = self.out_qdev_lower_layer1(q_deviation_lower_out)
    q_deviation_lower_out = self.relu_qdev_lower(q_deviation_lower_out)
    q_deviation_lower_out = self.out_qdev_lower_layer2(q_deviation_lower_out)
    q_deviation_lower_out = self.relu_qdev_lower(q_deviation_lower_out)
    q_deviation_lower_out = self.out_qdev_lower_layer3(q_deviation_lower_out)
    q_deviation_lower_out = torch.nn.functional.relu(q_deviation_lower_out) # truncate at zero, can't be negative
    
    if GLOBAL_ONLY:
      q_deviation_upper_out = self.out_qdev_upper_layer1(flow_global)
    else:      
      q_deviation_upper_out = self.qDevUpperHeadPremap(flow)
      q_deviation_upper_out = q_deviation_upper_out.reshape(-1, self.NUM_TOKENS_NET * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_QDEV)
      if self.global_dim > 0:
        q_deviation_upper_out = torch.concat([q_deviation_upper_out, flow_global], 1);       
      q_deviation_upper_out = self.out_qdev_upper_layer1(q_deviation_upper_out)
    q_deviation_upper_out = self.relu_qdev_upper(q_deviation_upper_out)
    q_deviation_upper_out = self.out_qdev_upper_layer2(q_deviation_upper_out)
    q_deviation_upper_out = self.relu_qdev_upper(q_deviation_upper_out)
    q_deviation_upper_out = self.out_qdev_upper_layer3(q_deviation_upper_out)
    q_deviation_upper_out = torch.nn.functional.relu(q_deviation_upper_out) # truncate at zero, can't be negative


    return policy_out, value_out, moves_left_out, unc_out, value2_out, q_deviation_lower_out, q_deviation_upper_out


  def compute_loss(self, loss_calc : LossCalculator, batch, policy_out, value_out, moves_left_out, unc_out,
                   value2_out, q_deviation_lower_out, q_deviation_upper_out, num_pos, last_lr, log_stats):
    policy_target = batch['policies']
    wdl_target = batch['wdl_result']
    q_target = batch['wdl_q']
    moves_left_target = batch['mlh']
    unc_target = batch['unc']
    wdl2_target = batch['wdl2_result']
    q_deviation_lower_target = batch['q_deviation_lower']
    q_deviation_upper_target = batch['q_deviation_upper']

    value_target = q_target * self.q_ratio + wdl_target * (1 - self.q_ratio)
    p_loss = loss_calc.policy_loss(policy_target, policy_out)
    v_loss = loss_calc.value_loss(value_target, value_out)
    v2_loss = loss_calc.value2_loss(wdl2_target, value2_out)
    ml_loss = loss_calc.moves_left_loss(moves_left_target, moves_left_out)
    u_loss = loss_calc.unc_loss(unc_target, unc_out)
    q_deviation_lower_loss = loss_calc.q_deviation_lower_loss(q_deviation_lower_target, q_deviation_lower_out)
    q_deviation_upper_loss = loss_calc.q_deviation_upper_loss(q_deviation_upper_target, q_deviation_upper_out)
    
    total_loss = (self.policy_loss_weight * p_loss
        + self.value_loss_weight * v_loss
        + self.moves_left_loss_weight * ml_loss
        + self.unc_loss_weight * u_loss
        + self.value2_loss_weight * v2_loss
        + self.q_deviation_loss_weight * q_deviation_lower_loss
        + self.q_deviation_loss_weight * q_deviation_upper_loss)
        
    if (log_stats):
        policy_accuracy = loss_calc.calc_accuracy(policy_target, policy_out, True)
        value_accuracy = loss_calc.calc_accuracy(value_target, value_out, False)
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
        self.fabric.log("q_deviation_lower_loss", q_deviation_lower_loss, step=num_pos)
        self.fabric.log("q_deviation_upper_loss", q_deviation_upper_loss, step=num_pos)
        
    return total_loss

