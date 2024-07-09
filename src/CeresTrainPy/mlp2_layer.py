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
from activation_functions import Swish, ReLUSquared


USE_BIAS = True # Daniel Moore reported biases useful in FFN


class MLP2Layer(torch.nn.Module):
  def __init__(self, model_dim: int, ffn_inner_dim: int, out_dim : int, activation_type : str, use_te : bool = False) -> None:
    super().__init__()
        
    self.activation_type = activation_type
    self.use_te = use_te

    if self.use_te:
      import transformer_engine.pytorch as te
      from transformer_engine.common.recipe import Format, DelayedScaling

      # TODO: Lift restriction that activation function must be 'gelu' for TE
      self.te_mlp_ln = te.LayerNormMLP(model_dim, ffn_inner_dim, bias=USE_BIAS, 
                                       return_layernorm_output = True, activation='gelu') 
      fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
      self.fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    
    else:
      self.linear1 = torch.nn.Linear(model_dim, ffn_inner_dim, bias=USE_BIAS)
      self.linear2 = torch.nn.Linear(ffn_inner_dim, out_dim, bias=USE_BIAS)
      if activation_type == 'SwiGLU':
        self.linear3 = torch.nn.Linear(model_dim, ffn_inner_dim, bias=False) 

    if (activation_type == 'None'):
      self.activation_fn = torch.nn.Identity()
    elif (activation_type == 'ReLU'):
      self.activation_fn = torch.nn.ReLU()
    elif (activation_type == 'ReLUSquared'):
      self.activation_fn = ReLUSquared()
    elif (activation_type == 'Swish'):
      self.activation_fn = Swish()
    elif (activation_type == 'Mish'):
      self.activation_fn = torch.nn.Mish()
    elif (activation_type == 'SwiGLU'):
      self.activation_fn = torch.nn.SiLU() # First of SwiGLU here
    else:
      raise Exception('Unknown activation type', activation_type)


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.use_te:
      with te.fp8_autocast(self.training, fp8_recipe=self.fp8_recipe):
        return self.te_mlp_ln(x) # TODO: figure out why returning a singleton here is required vs. tuple below
    else:      
      x = self.linear1(x)
      before_linear2 = self.activation_fn(x)
      if (self.activation_type == 'SwiGLU'):
          before_linear2 *= self.linear3(x)

      x_out = self.linear2(before_linear2)

    return before_linear2, x_out
