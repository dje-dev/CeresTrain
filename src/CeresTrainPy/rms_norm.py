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
from torch import Tensor

class RMSNorm(torch.nn.Module):
  def __init__(self, d_model, eps=1e-6):
    super().__init__()

    self.d_model = d_model
    self.eps = eps
    self.scale = torch.nn.Parameter(torch.ones(d_model))

  def forward(self, x : Tensor) -> Tensor:
      rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
      return x / rms * self.scale
