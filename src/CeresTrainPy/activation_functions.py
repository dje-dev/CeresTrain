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
