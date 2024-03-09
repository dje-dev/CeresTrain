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
from einops import einsum, rearrange
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
               global_stream_dim : int, num_attention_heads: int, kv_channels: int, norm_type : str, 
               layernorm_eps : float, attention_multiplier : int = 1,
               global_stream_attention_per_square : int = 0,
               smolgen_per_square_dim : int = 0, smolgen_intermediate_dim : int = 0, 
               smolgen_head_divisor : int = 1, smolgenPrepLayer = None,
               smolgen_activation_type : str = 'None', 
               use_rpe : bool = False,
               rpe_factor_q : torch.Tensor = None,
               rpe_factor_k : torch.Tensor = None,
               rpe_factor_v : torch.Tensor = None,
               transpose_out : bool = False,
               test : bool = False) -> None:
    super().__init__()

    self.num_tokens_q = num_tokens_q
    self.num_tokens_kv = num_tokens_kv
    self.global_stream_dim = global_stream_dim
    self.num_heads = num_attention_heads
    self.attention_multiplier = attention_multiplier
    self.d_model = num_attention_heads * kv_channels
    self.d_output = num_tokens_q if transpose_out else num_attention_heads * kv_channels
    self.d_k = kv_channels
    self.softmax = torch.nn.Softmax(-1)
    self.smolgen_head_divisor = smolgen_head_divisor
    self.global_stream_attention_per_square = global_stream_attention_per_square
    self.test = test    
    self.use_smolgen = smolgenPrepLayer is not None    
    self.use_rpe = use_rpe
    self.transpose_out = transpose_out  
    
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

    # If per-square attention is used, each square receives some per-square info from global stream and some shared from global stream
    # Othewise, a fully copy of the global stream is used for each square
    if self.global_stream_dim > 0:
      dim_global_info = self.global_stream_attention_per_square * 2 if global_stream_attention_per_square > 0 else self.global_stream_dim
    else:
      dim_global_info = 0  
      
    # Fused Q, K, and V linear projection for improved efficiency.
    self.qkv = torch.nn.Linear(self.d_model + dim_global_info, 3 * self.d_model * self.attention_multiplier, bias=USE_BIAS)
    out_width = self.num_tokens_q if self.transpose_out else self.d_output
    self.W_h = torch.nn.Linear(out_width * self.attention_multiplier, self.d_output)
#    print ('dout ', out_width * self.attention_multiplier, self.d_output)
    
    if self.use_rpe:
      assert num_tokens_q == num_tokens_kv, "RPE requires equal number of tokens for Q and K"     
      self.rpe_factor_q = LinearWrapper(rpe_factor_q)
      self.rpe_factor_k = LinearWrapper(rpe_factor_k)
#      self.rpe_factor_v = LinearWrapper(rpe_factor_v)
      
      RPE_INNER_DIM = 512
      self.rpe_q = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier, self.num_heads, RPE_INNER_DIM))
      self.rpe_k = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier, self.num_heads, RPE_INNER_DIM))
#      self.rpe_v = torch.nn.Parameter(torch.zeros(self.d_k, self.num_heads, RPE_INNER_DIM))

    self.smolgen_per_square_dim = smolgen_per_square_dim
    self.smolgen_intermediate_dim = smolgen_intermediate_dim


    if self.use_smolgen:
      self.wrapped_smolgen_prep_layer = LinearWrapper(smolgenPrepLayer) # wrap so shared layer is not re-registered
      self.sm1 = torch.nn.Linear(self.d_model, smolgen_per_square_dim)
      self.sm2 = torch.nn.Linear(num_tokens_q * smolgen_per_square_dim, smolgen_intermediate_dim)
      self.ln1 = torch.nn.LayerNorm(smolgen_intermediate_dim) if norm_type == 'LayerNorm' else RMSNorm(smolgen_intermediate_dim, eps=layernorm_eps)
      self.sm3 = torch.nn.Linear(smolgen_intermediate_dim, num_attention_heads * smolgen_intermediate_dim // smolgen_head_divisor)
      self.ln2 = torch.nn.LayerNorm(num_attention_heads * smolgen_intermediate_dim // smolgen_head_divisor) if norm_type == 'LayerNorm' else RMSNorm(num_attention_heads * smolgen_intermediate_dim// smolgen_head_divisor, eps=layernorm_eps)
      
    if global_stream_dim > 0 and self.global_stream_attention_per_square > 0:
      assert num_tokens_q == num_tokens_kv, "global_stream_attention_per_square requires equal number of tokens for Q and K"
      self.global_prep_attn_per_square = torch.nn.Linear(global_stream_dim, 64 * self.global_stream_attention_per_square, bias = False); 
      self.global_prep_attn            = torch.nn.Linear(global_stream_dim, self.global_stream_attention_per_square, bias = False); 
      

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

    if self.use_rpe:
      scores = scores + self.rpe_factor_q.linear(einsum(Q, self.rpe_q, "b h q d, d h z -> b h z")).reshape(-1, self.num_heads, self.num_tokens_q, self.num_tokens_q)
      scores = scores + self.rpe_factor_k.linear(einsum(K, self.rpe_k, "b h k d, d h z -> b h z")).reshape(-1, self.num_heads, self.num_tokens_q, self.num_tokens_q)
      # consider using scaling below as (3 * self.d_k) due to extra terms
      
    scores = scores / math.sqrt(self.d_k)

    if self.use_smolgen:
      assert self.num_tokens_q == self.num_tokens_kv, "use_smolgen requires equal number of tokens for Q and K"
      smolgen_logits_repeated = smolgen.reshape(smolgen.shape[0], self.num_heads, self.num_tokens_q, self.num_tokens_q)
      scores = scores + smolgen_logits_repeated
     
    A = self.softmax(scores)
#    print('A', A.shape) # 1024 16 64 64
    # Get the weighted average of the values
    H = torch.matmul(A, V)
#    print('H', H.shape) # 1024 16 64 24

#    if self.use_rpe:
#      H = H + einsum(A, self.rpe_v, "b h q k, d h q k -> b h q d") 

    return H, A
  

  def forward(self, x:torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, global_state : torch.Tensor) -> torch.Tensor:
    batch_size = query.size(0)

    if self.global_stream_dim > 0:
      if self.global_stream_attention_per_square > 0:
        # Compute and concatenate the per-square component
        global_state_for_attn_per_square = self.global_prep_attn_per_square(global_state)
        global_state_for_attn_per_square = global_state_for_attn_per_square.reshape([-1, 64, self.global_stream_attention_per_square])      
        qkv_x = torch.cat([query, global_state_for_attn_per_square], 2)
      
        # Compute and append the shared component
        global_state_for_attn = self.global_prep_attn(global_state)
        global_state_for_attn = global_state_for_attn.repeat(1,64).reshape([-1, 64, self.global_stream_attention_per_square])
        qkv_x = torch.concatenate([qkv_x, global_state_for_attn], 2)
      else:
        # append the full global state to the query (every square)
        global_state = global_state.repeat(1,self.num_tokens_q).reshape([-1, self.num_tokens_q, self.global_stream_dim])
        qkv_x = torch.cat([query, global_state], 2)
        
    else:
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
      H_cat, A = self.sdp_smolgen(Q, K, V, smolgen)
    else:
      if self.use_rpe:
        H_cat, A = self.sdp_smolgen(Q, K, V, None)
      else:
        H_cat = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    # Put all the heads back together by concat (with heads moved back to the right)
    H_cat =  H_cat.transpose(1, 2).contiguous().view(batch_size, -1, self.d_output * self.attention_multiplier)
      
    # Final linear layer  
    H = self.W_h(H_cat)

    return H
