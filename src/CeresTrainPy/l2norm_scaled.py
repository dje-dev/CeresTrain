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
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class L2NormScaled(Module):
  EPSILON = 1E-6

  def __init__(self, dim : int, with_scale : bool):
    """
    Initialize the LayerL2NormScaled module.
    
    :param dim: The dimension along which normalization is performed.
    :param with_scale: Boolean indicating whether scaling should be applied.
    """
    super().__init__()
    self.with_scale = with_scale
    self.dim = dim
    self.scale = nn.Parameter(torch.ones(1)) if with_scale else None

  def forward(self, tensor : Tensor) -> Tensor:
    """
    Perform forward pass for L2 normalization with optional scaling.

    :param tensor: Input tensor.
    :return: L2-normalized tensor with optional scaling.
    """
    normed = tensor * torch.rsqrt(tensor.pow(2).mean([self.dim], keepdim=True) + self.EPSILON)
    return normed * self.scale if self.with_scale else normed
