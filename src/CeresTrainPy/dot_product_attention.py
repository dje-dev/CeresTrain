# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

import math
import numpy as np

import torch

from einops import einsum, rearrange, repeat
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
  def __init__(self, num_tokens_q : int, num_tokens_kv : int,
               num_attention_heads: int, kv_channels: int, norm_type : str, 
               layernorm_eps : float, attention_multiplier : int = 1,
               smolgen_per_square_dim : int = 0, smolgen_intermediate_dim : int = 0, 
               smolgen_head_divisor : int = 1, smolgenPrepLayer = None,
               smolgen_activation_type : str = 'None', 
               use_rpe : bool = False,
               test : bool = False) -> None:
    super().__init__()

    self.num_tokens_q = num_tokens_q
    self.num_tokens_kv = num_tokens_kv
    self.num_heads = num_attention_heads
    self.attention_multiplier = attention_multiplier
    self.d_model = num_attention_heads * kv_channels
    self.d_output = num_attention_heads * kv_channels
    self.d_k = kv_channels
    self.softmax = torch.nn.Softmax(-1)
    self.smolgen_head_divisor = smolgen_head_divisor
    self.test = test    
    self.use_smolgen = smolgenPrepLayer is not None    
    self.use_rpe = use_rpe
    
    if self.use_smolgen:
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
    self.qkv = torch.nn.Linear(self.d_model, 3 * self.d_model * self.attention_multiplier, bias=USE_BIAS)
    self.W_h = torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_output)
    
    if self.use_rpe:
      self.rpe_factor = torch.nn.Parameter(make_rpe_map(), requires_grad=False)

      RPE_INNER_DIM = 16 # rounded up to power of 2 (there are only 15 possible values of a -  b where a and b are 0...7)
      self.rpe_q = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier * self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM))
      self.rpe_k = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier * self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM))
      self.rpe_v = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier * self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM))

    self.smolgen_per_square_dim = smolgen_per_square_dim
    self.smolgen_intermediate_dim = smolgen_intermediate_dim


    if self.use_smolgen:
      self.wrapped_smolgen_prep_layer = LinearWrapper(smolgenPrepLayer) # wrap so shared layer is not re-registered
      self.sm1 = torch.nn.Linear(self.d_model, smolgen_per_square_dim)
      self.sm2 = torch.nn.Linear(num_tokens_q * smolgen_per_square_dim, smolgen_intermediate_dim)
      self.ln1 = torch.nn.LayerNorm(smolgen_intermediate_dim) if norm_type == 'LayerNorm' else RMSNorm(smolgen_intermediate_dim, eps=layernorm_eps)
      self.sm3 = torch.nn.Linear(smolgen_intermediate_dim, num_attention_heads * smolgen_intermediate_dim // smolgen_head_divisor)
      self.ln2 = torch.nn.LayerNorm(num_attention_heads * smolgen_intermediate_dim // smolgen_head_divisor) if norm_type == 'LayerNorm' else RMSNorm(num_attention_heads * smolgen_intermediate_dim// smolgen_head_divisor, eps=layernorm_eps)
      

  @property
  def smolgenPrepLayer(self):
    return self.wrapped_smolgen_prep_layer.linear



  def sdp_and_smol_or_rpe(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, smolgen:torch.Tensor): # -> torch.Tensor, torch.Tensor:
    # Note that scaling could be done separately on Q and K to possibly improve stability. See:
    #   https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118
    #scaleDivisor = 1 # math.pow(self.d_k, 0.25) # apply sqrt twice since we are dividing twice
    #Q = Q / scaleDivisor
    #K = K / scaleDivisor
    scores = torch.matmul(Q, K.transpose(2, 3))

    if self.use_rpe:
      rpe_q = self.rpe_q @ self.rpe_factor
      rpe_q = rpe_q.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)

      rpe_k = self.rpe_k @ self.rpe_factor
      rpe_k = rpe_k.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)
      
      scores = scores + einsum(Q, rpe_q, "b h q d, d h q k->b h q k")
      scores = scores + einsum(K, rpe_k, "b h k d, d h q k->b h q k")
      # consider using scaling below as (3 * self.d_k) due to extra terms
      
    scores = scores / math.sqrt(self.d_k)

    if self.use_smolgen:
      assert self.num_tokens_q == self.num_tokens_kv, "use_smolgen requires equal number of tokens for Q and K"
      smolgen_logits_repeated = smolgen.reshape(smolgen.shape[0], self.num_heads, self.num_tokens_q, self.num_tokens_q)
      scores = scores + smolgen_logits_repeated
     
    A = self.softmax(scores)

    # Get the weighted average of the values
    H = torch.matmul(A, V)

    if self.use_rpe:
      rpe_v = self.rpe_v @ self.rpe_factor
      rpe_v = rpe_v.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)
      
      H = H + einsum(A, rpe_v, "b h q k, d h q k->b h q d")

    return H, A
  

  def forward(self, x:torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, global_state : torch.Tensor) -> torch.Tensor:
    batch_size = query.size(0)

    qkv_x = query    

    # Linear projections (Q, K, V jointly).
    qkv = self.qkv(qkv_x)

    # Split apart Q, K, V (with heads on the left)
    qkv = qkv.reshape(batch_size, -1, self.num_heads, 3*self.d_k * self.attention_multiplier)
    qkv = qkv.permute(0, 2, 1, 3)
    Q, K, V = qkv.chunk(3, dim=-1)
    
    if self.use_smolgen:
      smolgen = self.sm1(x)
      smolgen = smolgen.reshape(- 1, self.num_tokens_q * self.smolgen_per_square_dim)
      smolgen = self.sm2(smolgen)
      smolgen = self.smolgen_activation_fn(smolgen)
      smolgen = self.ln1(smolgen)
      smolgen = self.sm3(smolgen)
      smolgen = self.smolgen_activation_fn(smolgen)
      smolgen = self.ln2(smolgen)
      smolgen = smolgen.reshape(-1, self.num_heads, self.smolgen_intermediate_dim // self.smolgen_head_divisor)
      smolgen = self.smolgenPrepLayer(smolgen) # shared
      smolgen = smolgen.reshape(-1, self.num_heads, self.num_tokens_q, self.num_tokens_q)
      H_cat, A = self.sdp_and_smol_or_rpe(Q, K, V, smolgen)
    else:
      if self.use_rpe:
        H_cat, A = self.sdp_and_smol_or_rpe(Q, K, V, None)
      else:
        H_cat = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    # Put all the heads back together by concat (with heads moved back to the right)
    H_cat =  H_cat.transpose(1, 2).contiguous().view(batch_size, -1, self.d_output * self.attention_multiplier)
      
    # Final linear layer  
    H = self.W_h(H_cat)

    return H


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


