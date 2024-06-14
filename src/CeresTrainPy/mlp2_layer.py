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


class MLP2Layer(torch.nn.Module):
  def __init__(self, model_dim: int, ffn_inner_dim: int, out_dim : int, activation_type : str) -> None:
    super().__init__()
        
    self.activation_type = activation_type

    self.linear1 = torch.nn.Linear(model_dim, ffn_inner_dim, bias=False)
    self.linear2 = torch.nn.Linear(ffn_inner_dim, out_dim, bias=False)
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
    x = self.linear1(x)
    before_linear2 = self.activation_fn(x)
    if (self.activation_type == 'SwiGLU'):
        before_linear2 *= self.linear3(x)

    x_out = self.linear2(before_linear2)

    return before_linear2, x_out
