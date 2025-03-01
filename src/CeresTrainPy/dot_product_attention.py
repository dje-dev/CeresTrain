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


class ParameterWrapper:
  def __init__(self, parameter):
    self._parameter = parameter

  @property
  def parameter(self):
    return self._parameter


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
               layernorm_eps : float, 
               use_qkv : bool = True,
               attention_multiplier : int = 1,
               smolgen_per_square_dim : int = 0, smolgen_intermediate_dim : int = 0, 
               smolgen_head_divisor : int = 1, smolgenPrepLayer = None,
               smolgen_activation_type : str = 'None', 
               use_rpe : bool = False,
               use_rpe_v : bool = True,  
               rpe_factor_shared  = None,
               use_rel_bias: bool = False,
               use_nonlinear_attention: bool = False,
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
    self.use_qkv = use_qkv
    self.use_smolgen = smolgenPrepLayer is not None    
    self.use_rpe = use_rpe
    self.use_rpe_v = use_rpe_v
    self.use_rel_bias = use_rel_bias
    self.use_nonlinear_attention = use_nonlinear_attention  

    assert self.use_smolgen + self.use_rpe + self.use_rel_bias <= 1, "only one of smolgen, rpe, and rel bias can be enabled"
    
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
    self.qkv = torch.nn.Linear(self.d_model, 3 * self.d_model * self.attention_multiplier, bias = True if self.use_nonlinear_attention else USE_BIAS)
    self.W_h = torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_output)

    if self.use_nonlinear_attention:
      self.qkvLN = torch.nn.LayerNorm(self.d_model * self.attention_multiplier) if norm_type == 'LayerNorm' else RMSNorm(self.d_model * self.attention_multiplier)
      self.q2 = torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_model * self.attention_multiplier, bias=USE_BIAS)
      self.k2 = torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_model * self.attention_multiplier, bias=USE_BIAS)
      self.v2 = torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_model * self.attention_multiplier, bias=USE_BIAS)

      # extra layernorm for enahnced training stability
#      self.qLN = torch.nn.LayerNorm(self.d_model * self.attention_multiplier) if norm_type == 'LayerNorm' else RMSNorm(self.d_model * self.attention_multiplier)
#      self.kLN = torch.nn.LayerNorm(self.d_model * self.attention_multiplier) if norm_type == 'LayerNorm' else RMSNorm(self.d_model * self.attention_multiplier)

    RPE_INNER_DIM = 16 # rounded up to power of 2 (there are only 15 possible values of a -  b where a and b are 0...7)

    if self.use_rpe:
      self.wrapped_rpe_factor_shared = ParameterWrapper(rpe_factor_shared) # wrap so shared layer is not re-registered
      self.rpe_q = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier * self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM))
      self.rpe_k = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier * self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM))
      self.rpe_v = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier * self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM)) if self.use_rpe_v else None

      torch.nn.init.kaiming_uniform_(self.rpe_q, a=0.1)
      torch.nn.init.kaiming_uniform_(self.rpe_k, a=0.1)
      if self.use_rpe_v:
        torch.nn.init.kaiming_uniform_(self.rpe_v, a=0.1)

    if self.use_rel_bias:
      self.rel_bias = torch.nn.Parameter(torch.zeros(self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM))

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

  @property
  def rpeFactorShared(self):
    return self.wrapped_rpe_factor_shared.parameter.data

  # Function to cap logit scores (as used in the grok and gemma models).
  def soft_cap(self, score, softcap):
    score = score / softcap
    score = torch.tanh(score)
    score = score * softcap
    return score

 
  def sdp_and_smol_or_rpe(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, smolgen:torch.Tensor): # -> torch.Tensor, torch.Tensor:
    # Note that scaling could be done separately on Q and K to possibly improve stability. See:
    #   https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118
    #scaleDivisor = 1 # math.pow(self.d_k, 0.25) # apply sqrt twice since we are dividing twice
    #Q = Q / scaleDivisor
    #K = K / scaleDivisor
    scores = torch.matmul(Q, K.transpose(2, 3))

    if self.use_rpe:
      rpe_q = self.rpe_q @ self.rpeFactorShared
      rpe_q = rpe_q.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)

      rpe_k = self.rpe_k @ self.rpeFactorShared
      rpe_k = rpe_k.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)
      
      scores = scores + einsum(Q, rpe_q, "b h q d, d h q k->b h q k")
      scores = scores + einsum(K, rpe_k, "b h k d, d h q k->b h q k")
      # consider using scaling below as (3 * self.d_k) due to extra terms
       
    scores = scores / math.sqrt(self.d_k)

    if self.use_rel_bias:
      scores = scores + torch.reshape(self.rel_bias @ self.rpe_factor, [-1, 64, 64])

    if self.use_smolgen:
      assert self.num_tokens_q == self.num_tokens_kv, "use_smolgen requires equal number of tokens for Q and K"
      smolgen_logits_repeated = smolgen.reshape(smolgen.shape[0], self.num_heads, self.num_tokens_q, self.num_tokens_q)
      scores = scores + smolgen_logits_repeated
     
    # softcap logits for enhanced training stability
    SOFTCAP = 100
    scores = self.soft_cap(scores, SOFTCAP)

    A = self.softmax(scores)

    # Get the weighted average of the values
    H = torch.matmul(A, V)

    if self.use_rpe and self.use_rpe_v:
      rpe_v = self.rpe_v @ self.rpeFactorShared
      rpe_v = rpe_v.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)
      
      H = H + einsum(A, rpe_v, "b h q k, d h q k->b h q d")

    return H, A
  

  def calc_smolgen(self, x:torch.Tensor) -> torch.Tensor:
    smolgen = self.sm1(x)
    smolgen = smolgen.reshape(-1, self.num_tokens_q * self.smolgen_per_square_dim)

    smolgen = self.sm2(smolgen)
    smolgen = self.smolgen_activation_fn(smolgen)
    smolgen = self.ln1(smolgen)

    smolgen = self.sm3(smolgen)
    smolgen = self.smolgen_activation_fn(smolgen)
    smolgen = self.ln2(smolgen)

    smolgen = smolgen.reshape(-1, self.num_heads, self.smolgen_intermediate_dim // self.smolgen_head_divisor)
    smolgen = self.smolgenPrepLayer(smolgen)

    smolgen = smolgen.reshape(-1, self.num_heads, self.num_tokens_q, self.num_tokens_q)
    return smolgen


  def forward(self, x:torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    batch_size = query.size(0)

    qkv_x = query    

    # Linear projections (Q, K, V jointly).
    qkv = self.qkv(qkv_x)

    if not self.use_nonlinear_attention:
      # Split apart Q, K, V (with heads on the left)
      qkv = qkv.reshape(batch_size, -1, self.num_heads, 3*self.d_k * self.attention_multiplier)
      qkv = qkv.permute(0, 2, 1, 3)
      Q, K, V = qkv.chunk(3, dim=-1)    
    else:
      # Idea of introducing nonlinearity in the QKV was proposed in:
      #   "Neural Attention : Enhancing QKV Calculation in Self-Attention Mechanism with Neural Networks"
      #   Muhan Zhang, 2023
      qkv = qkv.reshape(batch_size, -1, 3, self.d_model * self.attention_multiplier)
      qkv = self.qkvLN(qkv)
      qkv = torch.nn.functional.mish(qkv)
      q, k, v = torch.unbind(qkv, dim=-2)

#      q = self.qLN(q)
#      k = self.kLN(k)

      Q = self.q2(q).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
      K = self.k2(k).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
      V = self.v2(v).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)  

    if self.use_smolgen:
      smolgen = self.calc_smolgen(x)
      H_cat, A = self.sdp_and_smol_or_rpe(Q, K, V, smolgen)
    else:
      if self.use_rpe:
        H_cat, A = self.sdp_and_smol_or_rpe(Q, K, V, None)
      else:
        # N.B. attention softcap is not implemented on this code path!
        H_cat = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    # Put all the heads back together by concat (with heads moved back to the right)
    H_cat =  H_cat.transpose(1, 2).contiguous().view(batch_size, -1, self.d_output * self.attention_multiplier)
      
    # Final linear layer  
    H = self.W_h(H_cat)

    return H



