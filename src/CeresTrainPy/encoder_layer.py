# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

from typing import Callable, Optional

import torch

from activation_functions import Swish, ReLUSquared
from rms_norm import RMSNorm
from soft_moe_batched_dual import SoftMoEBatchedDual
from mlp2_layer import MLP2Layer
from dot_product_attention import DotProductAttention


class EncoderLayer(torch.nn.Module):
  def __init__(self, trunk_type : str, 
               num_tokens_q : int, num_tokens_kv : int,
               num_layers: int, hidden_size: int, ffn_hidden_size: int, 
                num_attention_heads: int,  ffn_activation_type : str, norm_type : str, layernorm_eps : float = 1e-5, 
                smolgen_per_square_dim : int = 0, smolgen_intermediate_dim : int = 0, 
                smolgen_head_divisor : int = 1, smolgenPrepLayer = None,
                smolgen_activation_type : str = 'None',
                attention_multiplier : int = 1, 
                smoe_mode : str = 'None', smoe_num_experts : int = 0,
                alpha : float = 1, layerNum : int = 0, dropout_rate : float = 0,
                use_rpe : bool = False, 
                dual_attention_mode : str = 'None', test : bool = False):
    super().__init__()

    assert ffn_activation_type in ('ReLUSquared', 'ReLU', 'SwiGLU', 'Swish')
    assert norm_type in ('LayerNorm', 'RMSNorm') # None not supported

    self.trunk_type = trunk_type
    self.test = test
    self.layerNum = layerNum   
    self.numLayers = num_layers 
    self.alpha = alpha
    self.num_attention_heads = num_attention_heads
    self.dim_per_head = hidden_size // num_attention_heads
    self.attention_multiplier = attention_multiplier
    self.dropout_rate = dropout_rate
    self.dual_attention_mode = dual_attention_mode
    self.ffn_hidden_size = ffn_hidden_size
    
    self.ln1 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps) if norm_type == 'LayerNorm' else RMSNorm(hidden_size, eps=layernorm_eps)
    self.attention = DotProductAttention(num_tokens_q, num_tokens_kv,  num_attention_heads, self.dim_per_head, norm_type, layernorm_eps, 
                                         attention_multiplier, 
                                         smolgen_per_square_dim, smolgen_intermediate_dim, smolgen_head_divisor, smolgenPrepLayer, smolgen_activation_type, 
                                         use_rpe,  test)
    self.ln2 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps) if norm_type == 'LayerNorm' else RMSNorm(hidden_size, eps=layernorm_eps)

    if self.dual_attention_mode in ('DualAttentionAndFFN', 'DualAttentionOnly'):
      NUM_ATTENTION2_HEADS = 8 
      self.attention2 = DotProductAttention(hidden_size, hidden_size, NUM_ATTENTION2_HEADS, num_tokens_q//NUM_ATTENTION2_HEADS, norm_type, layernorm_eps, 
                                           1, 0, 0, 0, None, smolgen_activation_type, False, None, None, None, test)
      self.ln3 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps) if norm_type == 'LayerNorm' else RMSNorm(hidden_size, eps=layernorm_eps)
      if self.dual_attention_mode == 'DualAttentionAndFFN':
        self.mlp2 = MLP2Layer(model_dim=hidden_size, ffn_inner_dim=ffn_hidden_size, out_dim = hidden_size, activation_type=ffn_activation_type) 
        self.ln4 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps) if norm_type == 'LayerNorm' else RMSNorm(hidden_size, eps=layernorm_eps)

    if self.dropout_rate > 0:
      self.dropout_attn = torch.nn.Dropout(self.dropout_rate)
      self.dropout_mlp = torch.nn.Dropout(self.dropout_rate)

    SMOE_SLOTS_PER_EXPERT = 1
    SMOE_USE_NORMALIZATION = False
    SMOE_ONLY_SECOND_LAYER = smoe_mode in ('AddLinearSecondLayer', 'ReplaceLinearSecondLayer')
    SMOE_USE_BIAS = True
    self.smoe_mode = smoe_mode
    self.ffn_activation_type = ffn_activation_type
    
    if (smoe_num_experts > 0 and layerNum % 2 == 1):
      self.moe = SoftMoEBatchedDual(dim=hidden_size, ffn_dim=ffn_hidden_size,
                                    num_experts=smoe_num_experts, slots_per_expert=SMOE_SLOTS_PER_EXPERT,
                                    use_normalization=SMOE_USE_NORMALIZATION, only_second_layer=SMOE_ONLY_SECOND_LAYER,
                                    bias = SMOE_USE_BIAS) 
    else:
      self.moe = None

    if ffn_hidden_size > 0:
      self.mlp = MLP2Layer(model_dim=hidden_size, ffn_inner_dim=ffn_hidden_size, out_dim = hidden_size, activation_type=ffn_activation_type) 


  def forward(self, x: torch.Tensor, global_state : torch.Tensor) -> torch.Tensor:
    attn_output = self.attention(x, x, x, x, global_state)    
    
    if (self.dropout_rate > 0):
      attn_output = self.dropout_attn(attn_output)
      
    out1 = self.ln1(x * self.alpha + attn_output)

    if self.ffn_hidden_size > 0:
      if self.moe and self.smoe_mode == 'ReplaceLinear':
        assert self.ffn_activation_type in ('ReLUSquared') # SoftMoEBatchedDual currently only supports ReLUSquared
        mlp_output = self.moe(out1)
      else:
        (mlp_before_linear2, mlp_output) = self.mlp(out1) 
        if self.moe:
          if self.smoe_mode == 'AddLinearSecondLayer':
            mlp_output += self.moe(mlp_before_linear2)
          elif self.smoe_mode == 'ReplaceLinearSecondLayer':
            mlp_output = self.moe(mlp_before_linear2)
          else:
            assert False, f"Invalid smoe_mode {self.smoe_mode}"

        if self.dropout_rate > 0:
          mlp_output = self.dropout_mlp(mlp_output)
      
        out2 = self.ln2(out1 * self.alpha + mlp_output)
    else:
      out2 = out1
      
    if self.dual_attention_mode != 'None':
      out3_tr = out2.permute(0, 2, 1)
      attn_output3 = self.attention2(out3_tr, out3_tr, out3_tr, out3_tr, None)
      attn_output3 = attn_output3.permute(0, 2, 1)

      out3 = self.ln3(out2 * self.alpha + attn_output3)
      if self.dual_attention_mode == 'DualAttentionAndFFN':
        (_, mlp_output2) = self.mlp2(out3) 
        out3 = self.ln4(out3 * self.alpha + mlp_output2)
      out2 = out3
     
    return out2, None

