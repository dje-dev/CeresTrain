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
from activation_functions import to_activation
from rms_norm import RMSNorm

# An intuitive explanation of why biases are important can be found in 
# the YouTube video "How might LLMs store facts" by 3Blue1Brown (at about 9:00).
USE_BIAS = True # Daniel Moore reported biases useful in FFN

MLP_GLOBAL_PER_SQUARE_DIVISOR = 16; # reduces DIM ==> DIM / MLP_GLOBAL_PER_SQUARE_DIVISOR before flatten
MLP_GLOBAL_DIVISOR = 1; # divisor used to determine size of model dimension versus concatenated global dimension
MLP_GLOBAL_LN_EPS = 1e-6


class MLP2Layer(torch.nn.Module):
  def __init__(self, model_dim: int, ffn_inner_dim: int, out_dim : int, activation_type : str, norm_type : str, use_global : bool, use_te : bool = False) -> None:
    super().__init__()
        
    self.activation_type = activation_type
    self.use_te = use_te
    self.use_global = use_global

    if self.use_te:
      import transformer_engine.pytorch as te
      from transformer_engine.common.recipe import Format, DelayedScaling

      # TODO: Lift restriction that activation function must be 'gelu' for TE
      self.te_mlp_ln = te.LayerNormMLP(model_dim, ffn_inner_dim, bias=USE_BIAS, 
                                       return_layernorm_output = True, activation='gelu') 
      fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
      self.fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    
    else:
      if self.use_global:
        mlpGlobalPerSquare = model_dim // MLP_GLOBAL_PER_SQUARE_DIVISOR
        mlpGlobalDim = 64 * mlpGlobalPerSquare
        self.mlpGlobalSquareReduce = torch.nn.Linear(model_dim, mlpGlobalPerSquare, bias=USE_BIAS)
        self.mlpGlobalReduce = torch.nn.Linear(mlpGlobalDim, model_dim // MLP_GLOBAL_DIVISOR, bias=USE_BIAS)
        self.mlpGlobalLN = torch.nn.LayerNorm(model_dim // MLP_GLOBAL_DIVISOR, eps=MLP_GLOBAL_LN_EPS) if norm_type == 'LayerNorm' else RMSNorm(model_dim // MLP_GLOBAL_DIVISOR, eps=MLP_GLOBAL_LN_EPS)

      self.linear1 = torch.nn.Linear(model_dim + (model_dim // MLP_GLOBAL_DIVISOR if self.use_global else 0), ffn_inner_dim, bias=USE_BIAS)
      self.linear2 = torch.nn.Linear(ffn_inner_dim, out_dim, bias=USE_BIAS)
      if activation_type == 'SwiGLU':
        self.linear3 = torch.nn.Linear(model_dim, ffn_inner_dim, bias=False) 

    self.activation_fn = to_activation(activation_type)


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.use_te:
      with te.fp8_autocast(self.training, fp8_recipe=self.fp8_recipe):
        return self.te_mlp_ln(x) # TODO: figure out why returning a singleton here is required vs. tuple below
    else:
      if self.use_global:
        mlpGlobal = self.mlpGlobalSquareReduce(x);
        mlpGlobal = torch.flatten(mlpGlobal, 1);
        mlpGlobal = self.mlpGlobalReduce(mlpGlobal);
        mlpGlobal = self.activation_fn(mlpGlobal)
        mlpGlobal = self.mlpGlobalLN(mlpGlobal);
        x = torch.concat((x, mlpGlobal.unsqueeze(1).expand(-1, 64, -1)), dim=-1)

      x = self.linear1(x)
      before_linear2 = self.activation_fn(x)
      if (self.activation_type == 'SwiGLU'):
          before_linear2 *= self.linear3(x)

      x_out = self.linear2(before_linear2)

    return before_linear2, x_out
