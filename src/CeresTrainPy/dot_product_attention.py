# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

import torch
import math
from rms_norm import RMSNorm

from activation_functions import Swish, ReLUSquared

class LinearWrapper:
  def __init__(self, linear_layer):
    self._layer = linear_layer

  @property
  def linear(self):
    return self._layer


class DotProductAttention(torch.nn.Module):
  """
  Implements (scaled) Dot Product Attention.

  Parameters:
      num_attention_heads (int): Number of attention heads in the module.
      kv_channels (int): Number of channels (dimensions) in each key and value vector.
      norm_type (str): Type of normalization to apply within the attention mechanism.
      layernorm_eps (float): Epsilon value for layer normalization to prevent division by zero.
      attention_multiplier (int, optional): Scaling factor for attention scores. Defaults to 1.
      smolgen_per_square_dim (int, optional): Dimensionality for Smolgen per-square processing. Defaults to 0.
      smolgen_intermediate_dim (int, optional): Intermediate dimensionality for Smolgen processing. Defaults to 0.
      smolgenPrepLayer: Optional layer for preprocessing in the Smolgen context.
  """
  def __init__(self, num_attention_heads: int, kv_channels: int, norm_type : str, 
              layernorm_eps : float, attention_multiplier : int = 1,
              smolgen_per_square_dim : int = 0, smolgen_intermediate_dim : int = 0, 
              smolgen_head_divisor : int = 1, smolgenPrepLayer = None,
              smolgen_activation_type : str = 'None',
              test : bool = False) -> None:
    super().__init__()

    self.num_heads = num_attention_heads
    self.attention_multiplier = attention_multiplier
    self.d_model = num_attention_heads * kv_channels
    self.d_output = num_attention_heads * kv_channels
    self.d_k = kv_channels
    self.softmax = torch.nn.Softmax(-1)
    self.smolgen_head_divisor = smolgen_head_divisor
    self.test = test    
    
    if (smolgen_activation_type == 'None'):
      self.smolgen_activation_fn = torch.nn.Identity()
    elif (smolgen_activation_type == 'ReLU'):
      self.smolgen_activation_fn = torch.nn.ReLU()
    elif (smolgen_activation_type == 'ReLUSquared'):
      self.smolgen_activation_fn = ReLUSquared()
    elif (smolgen_activation_type == 'Swish'):
      self.smolgen_activation_fn = Swish()
    elif (smolgen_activation_type == 'SwiGLU'):
      self.smolgen_activation_fn = torch.nn.SiLU() # First of SwiGLU here
    else:
      raise Exception('Unknown activation type', smolgen_activation_type)


    # Implementations often but not always use no bias
    USE_BIAS = False

    # Fused Q, K, and V linear projection for improved efficiency.
    self.qkv = torch.nn.Linear(self.d_model, 3*self.d_model * self.attention_multiplier, bias=USE_BIAS)
    self.W_h = torch.nn.Linear(self.d_output * self.attention_multiplier, self.d_output)

    self.smolgen_per_square_dim = smolgen_per_square_dim
    self.smolgen_intermediate_dim = smolgen_intermediate_dim

    self.wrapped_smolgen_prep_layer = LinearWrapper(smolgenPrepLayer) # wrap so shared layer is not re-registered

    if smolgenPrepLayer is not None:
      self.sm1 = torch.nn.Linear(self.d_model, smolgen_per_square_dim)
      self.sm2 = torch.nn.Linear(64 * smolgen_per_square_dim, smolgen_intermediate_dim)
      self.ln1 = torch.nn.LayerNorm(smolgen_intermediate_dim) if norm_type == 'LayerNorm' else RMSNorm(smolgen_intermediate_dim, eps=layernorm_eps)
      self.sm3 = torch.nn.Linear(smolgen_intermediate_dim, num_attention_heads * smolgen_intermediate_dim // smolgen_head_divisor)
      self.ln2 = torch.nn.LayerNorm(num_attention_heads * smolgen_intermediate_dim // smolgen_head_divisor) if norm_type == 'LayerNorm' else RMSNorm(num_attention_heads * smolgen_intermediate_dim// smolgen_head_divisor, eps=layernorm_eps)
      
  @property
  def smolgenPrepLayer(self):
    return self.wrapped_smolgen_prep_layer.linear


  def sdp_smolgen(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, smolgen:torch.Tensor): # -> torch.Tensor, torch.Tensor:
    # Note that scaling could be done separately on Q and K to possibly improve stability. See:
    #   https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118
    #scaleDivisor = 1 # math.pow(self.d_k, 0.25) # apply sqrt twice since we are dividing twice
    #Q = Q / scaleDivisor
    #K = K / scaleDivisor
    scores = torch.matmul(Q, K.transpose(2, 3))
    scores = scores / math.sqrt(self.d_k)

    smolgen_logits_repeated = smolgen.reshape(smolgen.shape[0], self.num_heads, 64, 64)
    scores = scores + smolgen_logits_repeated
     
    A = self.softmax(scores)

    # Get the weighted average of the values
    H = torch.matmul(A, V)

    return H, A
  

  def forward(self, x:torch.Tensor, query: torch.Tensor,  key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    batch_size = query.size(0)

    # Linear projections (Q, K, V jointly).
    qkv = self.qkv(query)

    # Split apart Q, K, V (with heads on the left)
    qkv = qkv.reshape(batch_size, 64, self.num_heads, 3*self.d_k)
    qkv = qkv.permute(0, 2, 1, 3)
    Q, K, V = qkv.chunk(3, dim=-1)

    if self.smolgenPrepLayer is not None:
      smolgen = self.sm1(x)
      smolgen = smolgen.reshape(- 1, 64 * self.smolgen_per_square_dim)
      smolgen = self.sm2(smolgen)
      smolgen = self.smolgen_activation_fn(smolgen)
      smolgen = self.ln1(smolgen)
      smolgen = self.sm3(smolgen)
      smolgen = self.smolgen_activation_fn(smolgen)
      smolgen = self.ln2(smolgen)
      smolgen = smolgen.reshape(-1, self.num_heads, self.smolgen_intermediate_dim // self.smolgen_head_divisor)
      smolgen = self.smolgenPrepLayer(smolgen) # shared
      smolgen = smolgen.reshape(-1, self.num_heads, 64, 64)
      H_cat, A = self.sdp_smolgen(Q, K, V, smolgen)
    else:
      H_cat = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    # Put all the heads back together by concat (with heads moved back to the right)
    H_cat =  H_cat.transpose(1, 2).contiguous().view(batch_size, -1, self.d_output * self.attention_multiplier)

    # Final linear layer  
    H = self.W_h(H_cat)

    return H
