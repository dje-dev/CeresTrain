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


def to_activation(activation_str : str) -> torch.nn.Module:
  """
  Converts a string identifier of activation to a PyTorch activation function.
  """
  if activation_str == 'ReLU':
    return torch.nn.ReLU()
  elif activation_str == 'ReLUSquared':
    return ReLUSquared()
  elif activation_str == 'Swish':
    return Swish()
  elif activation_str == 'Mish':
    return torch.nn.Mish()
  elif activation_str == 'None' or activation_str == 'Identity':
    return torch.nn.Identity()
  elif activation_str == 'SwiGLU':
    assert(False, "SwiGLU disabled. Use requires two functions, SiLU (here) but also subsequent Linear (see mlp2 for example)")
    #self.activation_fn = torch.nn.SiLU() # First 
  else:
    raise Exception('Unknown activation type', activation_str)
 

class Swish(torch.nn.Module):
  """
  Swish activation function.
  Applies the Swish function element-wise:
  Swish(x) = x * sigmoid(x)
  """
  def __init__(self):
      super().__init__()
      
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return x * torch.sigmoid(x)


class ReLUSquared(torch.nn.Module):
  """
  ReLU-Squared activation function, as described in "Primer: Searching for Efficient Transformers for Language Modeling".
  Applies the ReLU function followed by squaring element-wise:
  ReLUSquared(x) = relu(x)^2
  """
  def __init__(self):
    super().__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()
