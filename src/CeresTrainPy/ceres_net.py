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
        q_ratio: Fraction of weight to be put on Q (search value results) versus game WDL result.
        optimizer: Optimizer to be used for training.
        learning_rate: Learning rate for the optimizer.
    """
    super().__init__()

    self.fabric = fabric
    self.save_hyperparameters()

    self.MAX_MOVES = 1858
    self.NUM_INPUT_BYTES_PER_SQUARE = 135 # N.B. also update in new.py
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

    self.embedding_layer = nn.Linear(in_features=self.NUM_INPUT_BYTES_PER_SQUARE, out_features=self.EMBEDDING_DIM )

    HEAD_MULT = config.NetDef_HeadWidthMultiplier

    self.HEAD_PREMAP_DIVISOR_POLICY = 8
    FINAL_POLICY_FC1_SIZE = 128 * HEAD_MULT
    FINAL_POLICY_FC2_SIZE = 64 * HEAD_MULT
    self.policyHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY)

    self.fcPolicyFinal1 = nn.Linear(self.NUM_TOKENS * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY, FINAL_POLICY_FC1_SIZE)
    self.fcPolicyRELU1 = self.Activation
    self.fcPolicyFinal2 = nn.Linear(FINAL_POLICY_FC1_SIZE, FINAL_POLICY_FC2_SIZE)
    self.fcPolicyRELU2 = self.Activation
    self.fcPolicyFinal3 = nn.Linear(FINAL_POLICY_FC2_SIZE, 1858)

    self.HEAD_PREMAP_DIVISOR_VALUE = 8
    FINAL_VALUE_FC1_SIZE = 32 * HEAD_MULT
    FINAL_VALUE_FC2_SIZE = 8 * HEAD_MULT
    self.valueHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY)
    self.out_value_layer1 = nn.Linear(in_features=self.NUM_TOKENS * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_VALUE,out_features=FINAL_VALUE_FC1_SIZE)
    self.relu_value1 = self.Activation
    self.out_value_layer2 = nn.Linear(in_features=FINAL_VALUE_FC1_SIZE, out_features=FINAL_VALUE_FC2_SIZE)
    self.relu_value2 = self.Activation 
    self.out_value_layer3 = nn.Linear(in_features=FINAL_VALUE_FC2_SIZE,out_features=3)

    self.HEAD_PREMAP_DIVISOR_MLH = 16
    FINAL_MLH_FC1_SIZE = 32 * HEAD_MULT
    FINAL_MLH_FC2_SIZE = 8 * HEAD_MULT
    self.mlhHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_MLH)
    self.out_mlh_layer1 = nn.Linear(in_features=self.NUM_TOKENS * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_MLH,out_features=FINAL_MLH_FC1_SIZE)
    self.relu_mlh = self.Activation
    self.out_mlh_layer2 = nn.Linear(in_features=FINAL_MLH_FC1_SIZE,out_features=FINAL_MLH_FC2_SIZE)
    self.relu_mlh = self.Activation 
    self.out_mlh_layer3 = nn.Linear(in_features=FINAL_MLH_FC2_SIZE,out_features=1)


    self.HEAD_PREMAP_DIVISOR_UNC = 16
    FINAL_UNC_FC1_SIZE = 32 * HEAD_MULT
    FINAL_UNC_FC2_SIZE = 8 * HEAD_MULT
    self.uncHeadPremap = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_UNC)
    self.out_unc_layer1 = nn.Linear(in_features=self.NUM_TOKENS * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_UNC,out_features=FINAL_UNC_FC1_SIZE)
    self.relu_unc = self.Activation
    self.out_unc_layer2 = nn.Linear(in_features=FINAL_UNC_FC1_SIZE,out_features=FINAL_UNC_FC2_SIZE)
    self.relu_unc = self.Activation 
    self.out_unc_layer3 = nn.Linear(in_features=FINAL_UNC_FC2_SIZE,out_features=1)

    self.DEEPNORM = config.NetDef_DeepNorm

    if self.DEEPNORM:     
      self.alpha = math.pow(2 * self.NUM_LAYERS, 0.25)
    else:      
      self.alpha = 1

    SMOLGEN_PER_SQUARE_DIM = config.NetDef_SmolgenDimPerSquare
    SMOLGEN_INTERMEDIATE_DIM = config.NetDef_SmolgenDim

    ATTENTION_MULTIPLIER = config.NetDef_AttentionMultiplier

    if config.NetDef_SoftMoE_NumExperts > 0:
      assert config.NetDef_SoftMoE_MoEMode == "AddLinearSecondLayer", 'implementation restriction: only AddLinearSecondLayer currently supported'
      assert config.NetDef_SoftMoE_NumSlotsPerExpert == 1
      assert config.NetDef_SoftMoE_UseBias == True
      assert config.NetDef_SoftMoE_UseNormalization == False
      assert config.NetDef_SoftMoE_OnlyForAlternatingLayers == True
      
    EPS = 1E-6
    
    if SMOLGEN_PER_SQUARE_DIM > 0:
      self.smolgenPrepLayer = nn.Linear(self.NUM_HEADS * SMOLGEN_INTERMEDIATE_DIM, 64 * 64)
    else:
      self.smolgenPrepLayer = None

    self.transformer_layer = torch.nn.Sequential(*[EncoderLayer(self.NUM_LAYERS, self.EMBEDDING_DIM, self.FFN_MULT*self.EMBEDDING_DIM, self.NUM_HEADS, 
                                                                ffn_activation_type = config.NetDef_FFNActivationType, 
                                                                norm_type = config.NetDef_NormType, layernorm_eps=EPS, 
                                                                attention_multiplier = ATTENTION_MULTIPLIER,
                                                                smoe_num_experts = config.NetDef_SoftMoE_NumExperts,
                                                                smolgen_per_square_dim = SMOLGEN_PER_SQUARE_DIM, 
                                                                smolgen_intermediate_dim = SMOLGEN_INTERMEDIATE_DIM, 
                                                                smolgenPrepLayer = self.smolgenPrepLayer, 
                                                                alpha=self.alpha, layerNum=i, dropout_rate=self.DROPOUT_RATE)
                                                  for i in range(self.NUM_LAYERS)])
      
    self.policy_loss_weight = policy_loss_weight
    self.value_loss_weight = value_loss_weight
    self.moves_left_loss_weight = moves_left_loss_weight
    self.unc_loss_weight = unc_loss_weight
    self.q_ratio = q_ratio
    self.optimizer = optimizer
    self.learning_rate = learning_rate


  def forward(self, input_planes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(input_planes, list):
      # when saving/restoring from ONNX the input will appear as a list instead of sequence of arguments
      input_planes = input_planes[0]

    flow = input_planes

    # Embedding layer.
    flow = flow.reshape(-1, self.NUM_TOKENS, (64 * self.NUM_INPUT_BYTES_PER_SQUARE) // self.NUM_TOKENS)
    flow = self.embedding_layer(flow)

    # Main transformer body.
    flow = self.transformer_layer(flow)

    # Heads.
    headOut = self.policyHeadPremap(flow)
    flattenedPolicy = headOut.reshape(-1, self.NUM_TOKENS * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_POLICY)
    ff1Policy = self.fcPolicyFinal1(flattenedPolicy)
    ff1RELUPolicy = self.fcPolicyRELU1(ff1Policy)
    ff2Policy = self.fcPolicyFinal2(ff1RELUPolicy)
    ff2RELUPolicy = self.fcPolicyRELU2(ff2Policy)
    policy_out = self.fcPolicyFinal3(ff2RELUPolicy)

    value_out = self.valueHeadPremap(flow)
    value_out = value_out.reshape(-1, self.NUM_TOKENS * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_VALUE)
    value_out = self.out_value_layer1(value_out)
    value_out = self.relu_value1(value_out)
    value_out = self.out_value_layer2(value_out)
    value_out = self.relu_value1(value_out)
    value_out = self.out_value_layer3(value_out)

    moves_left_out = self.mlhHeadPremap(flow)
    moves_left_out = moves_left_out.reshape(-1, self.NUM_TOKENS * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_MLH)
    moves_left_out = self.out_mlh_layer1(moves_left_out)
    moves_left_out = self.relu_mlh(moves_left_out)
    moves_left_out = self.out_mlh_layer2(moves_left_out)
    moves_left_out = self.relu_mlh(moves_left_out)
    moves_left_out = self.out_mlh_layer3(moves_left_out)
    moves_left_out = torch.nn.functional.relu(moves_left_out) # truncate at zero, can't be negative

    unc_out = self.uncHeadPremap(flow)
    unc_out = unc_out.reshape(-1, self.NUM_TOKENS * self.EMBEDDING_DIM // self.HEAD_PREMAP_DIVISOR_UNC)
    unc_out = self.out_unc_layer1(unc_out)
    unc_out = self.relu_unc(unc_out)
    unc_out = self.out_unc_layer2(unc_out)
    unc_out = self.relu_unc(unc_out)
    unc_out = self.out_unc_layer3(unc_out)
    unc_out = torch.nn.functional.relu(unc_out) # truncate at zero, can't be negative

    return policy_out, value_out, moves_left_out, unc_out


  def compute_loss(self, loss_calc : LossCalculator, batch, policy_out, value_out, moves_left_out, unc_out, num_pos, last_lr, log_stats):
    policy_target = batch['policies']
    wdl_target = batch['wdl_result']
    q_target = batch['wdl_q']
    moves_left_target = batch['mlh']
    unc_target = batch['unc']

    value_target = q_target * self.q_ratio + wdl_target * (1 - self.q_ratio)
    p_loss = loss_calc.policy_loss(policy_target, policy_out)
    v_loss = loss_calc.value_loss(value_target, value_out)
    ml_loss = loss_calc.moves_left_loss(moves_left_target, moves_left_out)
    u_loss = loss_calc.unc_loss(unc_target, unc_out)
    total_loss = (self.policy_loss_weight * p_loss
        + self.value_loss_weight * v_loss
        + self.moves_left_loss_weight * ml_loss
        + self.unc_loss_weight * u_loss)
        
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
        self.fabric.log("moves_left_loss", ml_loss, step=num_pos)
        self.fabric.log("unc_loss", u_loss, step=num_pos)
    return total_loss

