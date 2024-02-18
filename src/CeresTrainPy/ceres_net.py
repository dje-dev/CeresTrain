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
    self.NUM_TOKENS = 64 # N.B. ALSO UPDATE IN new.py
  
    self.DROPOUT_RATE = config.Exec_DropoutRate
    self.EMBEDDING_DIM = config.NetDef_ModelDim
    self.NUM_LAYERS = config.NetDef_NumLayers


    self.TRANSFORMER_OUT_DIM = self.EMBEDDING_DIM * self.NUM_TOKENS

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

    self.embedding_layer = nn.Linear(self.NUM_INPUT_BYTES_PER_SQUARE, self.EMBEDDING_DIM )
    self.embedding_layer_global = nn.Linear(64 * self.NUM_INPUT_BYTES_PER_SQUARE, config.NetDef_GlobalStreamDim)
    self.global_dim = config.NetDef_GlobalStreamDim
    
    HEAD_MULT = config.NetDef_HeadWidthMultiplier

    self.HEAD_PREMAP_DIVISOR_POLICY = 8
    FINAL_POLICY_FC1_SIZE = 128 * HEAD_MULT
    FINAL_POLICY_FC2_SIZE = 64 * HEAD_MULT
    self.policyHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY)

    self.fcPolicyFinal1 = nn.Linear(config.NetDef_GlobalStreamDim + self.NUM_TOKENS * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY, FINAL_POLICY_FC1_SIZE)
    self.fcPolicyRELU1 = self.Activation
    self.fcPolicyFinal2 = nn.Linear(FINAL_POLICY_FC1_SIZE, FINAL_POLICY_FC2_SIZE)
    self.fcPolicyRELU2 = self.Activation
    self.fcPolicyFinal3 = nn.Linear(FINAL_POLICY_FC2_SIZE, 1858)

    self.HEAD_PREMAP_DIVISOR_VALUE = 8
    FINAL_VALUE_FC1_SIZE = 32 * HEAD_MULT
    FINAL_VALUE_FC2_SIZE = 8 * HEAD_MULT

    self.out_value_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_VALUE_FC1_SIZE)
    self.relu_value_1 = self.Activation
    self.out_value_layer2 = nn.Linear(FINAL_VALUE_FC1_SIZE, FINAL_VALUE_FC2_SIZE)
    self.relu_value_2 = self.Activation 
    self.out_value_layer3 = nn.Linear(FINAL_VALUE_FC2_SIZE, 3)

    self.out_value2_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_VALUE_FC1_SIZE)
    self.relu_value2_1 = self.Activation
    self.out_value2_layer2 = nn.Linear(FINAL_VALUE_FC1_SIZE, FINAL_VALUE_FC2_SIZE)
    self.relu_value2_2 = self.Activation 
    self.out_value2_layer3 = nn.Linear(FINAL_VALUE_FC2_SIZE,3)

    self.HEAD_PREMAP_DIVISOR_MLH = 16
    FINAL_MLH_FC1_SIZE = 32 * HEAD_MULT
    FINAL_MLH_FC2_SIZE = 8 * HEAD_MULT
    self.out_mlh_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_MLH_FC1_SIZE)
    self.relu_mlh = self.Activation
    self.out_mlh_layer2 = nn.Linear(FINAL_MLH_FC1_SIZE, FINAL_MLH_FC2_SIZE)
    self.out_mlh_layer3 = nn.Linear(FINAL_MLH_FC2_SIZE, 1)

    self.HEAD_PREMAP_DIVISOR_UNC = 16
    FINAL_UNC_FC1_SIZE = 32 * HEAD_MULT
    FINAL_UNC_FC2_SIZE = 8 * HEAD_MULT
    self.out_unc_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_UNC_FC1_SIZE)
    self.relu_unc = self.Activation
    self.out_unc_layer2 = nn.Linear(FINAL_UNC_FC1_SIZE, FINAL_UNC_FC2_SIZE)
    self.out_unc_layer3 = nn.Linear(FINAL_UNC_FC2_SIZE, 1)

    self.HEAD_PREMAP_DIVISOR_QDEV = 16
    FINAL_QDEV_FC1_SIZE = 32 * HEAD_MULT
    FINAL_QDEV_FC2_SIZE = 8 * HEAD_MULT

    self.out_qdev_lower_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_QDEV_FC1_SIZE)
    self.relu_qdev_lower = self.Activation
    self.out_qdev_lower_layer2 = nn.Linear(FINAL_QDEV_FC1_SIZE,FINAL_QDEV_FC2_SIZE)
    self.out_qdev_lower_layer3 = nn.Linear(FINAL_QDEV_FC2_SIZE, 1)

    self.out_qdev_upper_layer1 = nn.Linear(config.NetDef_GlobalStreamDim, FINAL_QDEV_FC1_SIZE)
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
    
    if SMOLGEN_PER_SQUARE_DIM > 0:
      self.smolgenPrepLayer = nn.Linear(SMOLGEN_INTERMEDIATE_DIM // config.NetDef_SmolgenToHeadDivisor, 64 * 64)
    else:
      self.smolgenPrepLayer = None

    self.transformer_layer = torch.nn.Sequential(*[EncoderLayer(self.NUM_LAYERS, self.EMBEDDING_DIM, config.NetDef_GlobalStreamDim,
                                                                self.FFN_MULT*self.EMBEDDING_DIM, self.NUM_HEADS, 
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
                                                                test = config.Exec_TestFlag)
                                                  for i in range(self.NUM_LAYERS)])

    if config.NetDef_GlobalStreamDim > 0:
      NUM_GLOBAL_INNER = 512
      self.global_ffn1 = torch.nn.Sequential(*[nn.Linear(config.NetDef_GlobalStreamDim + 64 * 16, NUM_GLOBAL_INNER) for i in range(self.NUM_LAYERS)])
      self.global_ffn2 = torch.nn.Sequential(*[nn.Linear(NUM_GLOBAL_INNER, config.NetDef_GlobalStreamDim)for i in range(self.NUM_LAYERS)])

      # translate next 2 lines         
#            layersGlobalStreamFFN1[layerNum] = Linear(TransformerConfig.GlobalStreamDim + (64 * NetTransformer
#                                                     LayerEncoder.PER_SQUARE_REDUCED_DIM_TO_GLOBAL_STREAM), 1024, true, ExecutionConfig.Device, ExecutionConfig.DataType);
#            layersGlobalStreamFFN2[layerNum] = Linear(1024, TransformerConfig.GlobalStreamDim, true, ExecutionConfig.Device, ExecutionConfig.DataType);
          
      
    self.policy_loss_weight = policy_loss_weight
    self.value_loss_weight = value_loss_weight
    self.moves_left_loss_weight = moves_left_loss_weight
    self.unc_loss_weight = unc_loss_weight
    self.value2_loss_weight = value2_loss_weight
    self.q_deviation_loss_weight = q_deviation_loss_weight
    self.q_ratio = q_ratio
    self.optimizer = optimizer
    self.learning_rate = learning_rate


  def forward(self, input_planes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(input_planes, list):
      # when saving/restoring from ONNX the input will appear as a list instead of sequence of arguments
      input_planes = input_planes[0]

    flow = input_planes

    # Embedding layer.
    flow = flow.reshape(-1, self.NUM_TOKENS, (64 * self.NUM_INPUT_BYTES_PER_SQUARE) // self.NUM_TOKENS)
    flow = self.embedding_layer(flow)

    flow_global = input_planes.reshape(-1, 64 * self.NUM_INPUT_BYTES_PER_SQUARE)
    flow_global = self.embedding_layer_global(flow_global)
    
    # Main transformer body (stack of encoder layers)
    for i in range(self.NUM_LAYERS):
      flow, global_update = self.transformer_layer[i](flow, flow_global)
 
      # Update state based on the output of the encoder layer. 
      if self.global_dim > 0:# &&  NetTransformerLayerEncoder.PER_SQUARE_REDUCED_DIM_TO_GLOBAL_STREAM > 0)  
        global_update = global_update.reshape(-1, 64 * 16)     
 
        flow_state_input = torch.cat([flow_global, global_update], 1)  
      else:
        flow_state_input = flow
        
      if self.global_dim > 0:
        flow_global = self.global_ffn1[i](flow_state_input)
        flow_global = torch.nn.functional.relu(flow_global)
        #flow_global = self.Activation(flow_global)
        flow_global = self.global_ffn2[i](flow_global)
      

    # Heads.
    headOut = self.policyHeadPremap(flow)
    flattenedPolicy = headOut.reshape(-1, self.NUM_TOKENS * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY)
    if self.global_dim > 0:
      flattenedPolicy = torch.concat([flattenedPolicy, flow_global], 1);       

    ff1Policy = self.fcPolicyFinal1(flattenedPolicy)
    ff1RELUPolicy = self.fcPolicyRELU1(ff1Policy)
    ff2Policy = self.fcPolicyFinal2(ff1RELUPolicy)
    ff2RELUPolicy = self.fcPolicyRELU2(ff2Policy)
    policy_out = self.fcPolicyFinal3(ff2RELUPolicy)

    value_out = self.out_value_layer1(flow_global)
    value_out = self.relu_value_1(value_out)
    value_out = self.out_value_layer2(value_out)
    value_out = self.relu_value_2(value_out)
    value_out = self.out_value_layer3(value_out)

    value2_out = self.out_value2_layer1(flow_global)
    value2_out = self.relu_value2_1(value2_out)
    value2_out = self.out_value2_layer2(value2_out)
    value2_out = self.relu_value2_2(value2_out)
    value2_out = self.out_value2_layer3(value2_out)

    moves_left_out = self.out_mlh_layer1(flow_global)
    moves_left_out = self.relu_mlh(moves_left_out)
    moves_left_out = self.out_mlh_layer2(moves_left_out)
    moves_left_out = self.relu_mlh(moves_left_out)
    moves_left_out = self.out_mlh_layer3(moves_left_out)
    moves_left_out = torch.nn.functional.relu(moves_left_out) # truncate at zero, can't be negative

    unc_out = self.out_unc_layer1(flow_global)
    unc_out = self.relu_unc(unc_out)
    unc_out = self.out_unc_layer2(unc_out)
    unc_out = self.relu_unc(unc_out)
    unc_out = self.out_unc_layer3(unc_out)
    unc_out = torch.nn.functional.relu(unc_out) # truncate at zero, can't be negative

    q_deviation_lower_out = self.out_qdev_lower_layer1(flow_global)
    q_deviation_lower_out = self.relu_qdev_lower(q_deviation_lower_out)
    q_deviation_lower_out = self.out_qdev_lower_layer2(q_deviation_lower_out)
    q_deviation_lower_out = self.relu_qdev_lower(q_deviation_lower_out)
    q_deviation_lower_out = self.out_qdev_lower_layer3(q_deviation_lower_out)
    q_deviation_lower_out = torch.nn.functional.relu(q_deviation_lower_out) # truncate at zero, can't be negative
    
    q_deviation_upper_out = self.out_qdev_upper_layer1(flow_global)
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

