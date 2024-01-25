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
  def __init__(self, num_layers: int, hidden_size: int, ffn_hidden_size: int, 
                num_attention_heads: int,  ffn_activation_type : str, norm_type : str, layernorm_eps : float = 1e-5, 
                smolgen_per_square_dim : int = 0, smolgen_intermediate_dim : int = 0, smolgenPrepLayer = None,
                attention_multiplier : int = 1, smoe_mode : str = 'None', smoe_num_experts : int = 0,
                alpha : float = 1, layerNum : int = 0, dropout_rate : float = 0):
    super().__init__()

    assert ffn_activation_type in ('ReLUSquared', 'ReLU', 'SwiGLU', 'Swish')
    assert norm_type in ('LayerNorm', 'RMSNorm') # None not supported

    self.layerNum = layerNum   
    self.numLayers = num_layers 
    self.alpha = alpha
    self.num_attention_heads = num_attention_heads
    self.dim_per_head = hidden_size // num_attention_heads
    self.attention_multiplier = attention_multiplier
    self.dropout_rate = dropout_rate
    self.ln1 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps) if norm_type == 'LayerNorm' else RMSNorm(hidden_size, eps=layernorm_eps)
    self.attention = DotProductAttention(num_attention_heads, self.dim_per_head, norm_type, layernorm_eps, attention_multiplier, smolgen_per_square_dim, smolgen_intermediate_dim, smolgenPrepLayer)
    self.ln2 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps) if norm_type == 'LayerNorm' else RMSNorm(hidden_size, eps=layernorm_eps)

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

    self.mlp = MLP2Layer(model_dim=hidden_size, ffn_inner_dim=ffn_hidden_size, activation_type=ffn_activation_type) 


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    attn_output = self.attention.forward(x, x, x, x)
        
    if (self.dropout_rate > 0):
      attn_output = self.dropout_attn(attn_output)

    out1 = self.ln1(x * self.alpha + attn_output)
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

    return out2

    # FUNCTIONAL postnorm emultaes C# with 28% faster and slightly better accuracy out to 
    res = x
    
    # Fused QKV projection
    qkv = self.qkv_projection(x)
    qkv = qkv.view(qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.dim_per_head)
    q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

    # mha        
    x = self.attention(q, k, v)
#        x = self.ln2(x)
    x = self.projection(x)
    
    x = self.ln1(res +  x)
    res = x
    #x = self.ln2(x)
    x = self.mlp(x)
    
    return self.ln2(res + x)
