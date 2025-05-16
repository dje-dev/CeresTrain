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
  elif activation_str == 'APTx':
    return APTx(alpha=1.0, beta=1.0, gamma=0.5, trainable=False)
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



class APTx(torch.nn.Module):
  r"""The APTx (Alpha Plus Tanh Times) activation function: 
  Research Paper:: APTx: Better Activation Function than MISH, SWISH, and ReLU's Variants used in Deep Learning
  DOI Link: https://doi.org/10.51483/IJAIML.2.2.2022.56-61
  Arxiv: https://arxiv.org/abs/2209.06119
  Copied from: https://github.com/mr-ravin/aptx_activation/blob/main/aptx_activation/__init__.py
  .. math::
      \mathrm{APTx}(x) = (\alpha + \tanh(\beta x)) \cdot \gamma x
    
  :param alpha: Initial alpha value (default: 1.0)
  :param beta: Initial beta value (default: 1.0)
  :param gamma: Initial gamma value (default: 0.5)
  :param trainable: If True, all parameters (alpha, beta, gamma) become learnable (default: False)
  """
  def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, trainable=False):
    super().__init__()
        
    # Convert to tensors first
    alpha = torch.as_tensor(float(alpha))
    beta = torch.as_tensor(float(beta))
    gamma = torch.as_tensor(float(gamma))

    if trainable:
      self.alpha = torch.nn.Parameter(alpha)
      self.beta = torch.nn.Parameter(beta)
      self.gamma = torch.nn.Parameter(gamma)
    else:
      self.register_buffer("alpha", alpha)
      self.register_buffer("beta", beta)
      self.register_buffer("gamma", gamma)

  def forward(self, x):
    """Forward pass"""
    return (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x

  def extra_repr(self):
    """Show trainable status in string representation"""
    params = []
    for name in ["alpha", "beta", "gamma"]:
      tensor = getattr(self, name)
      if isinstance(tensor, torch.nn.Parameter):
        params.append(f"{name}=TRAIN")
      else:
        params.append(f"{name}={tensor.item():.2f}")
    return ", ".join(params)
  