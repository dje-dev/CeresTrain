# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

# based on: https://github.com/fkodom/soft-mixture-of-experts/blob/main/soft_mixture_of_experts/multi_expert.py

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from einops import einsum, rearrange
from torch import Tensor, nn

from multi_expert import MultiExpertLayer
from l2norm_scaled import L2NormScaled


class SoftMoEBatchedDual(nn.Module):
    """A PyTorch module for Soft-MoE, as described in the paper:
        "From Sparse to Soft Mixtures of Experts"
        https://arxiv.org/pdf/2308.00951.pdf

    einstein notation:
    - b: batch size
    - m: input sequence length
    - d: embedding dimension
    - n: num experts
    - p: num slots per expert
    - (n * p): total number of slots

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        slots_per_expert (int): number of slots per expert (p)
        bias (bool): whether to include a bias term. Default: True.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_experts: int,
        slots_per_expert: int,
        use_normalization : bool,
        only_second_layer : bool,
        bias: bool = True):

        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.use_normalization = use_normalization
        self.only_second_layer = only_second_layer
        self.bias = bias

        dim_dispatch_in = ffn_dim if only_second_layer else dim
        self.phi = nn.Parameter(torch.empty((dim_dispatch_in, num_experts, slots_per_expert)))

        if not self.only_second_layer:
          self.experts1 = MultiExpertLayer(dim, ffn_dim, num_experts, bias)
        
        self.experts2 = MultiExpertLayer(ffn_dim, dim, num_experts, bias)

        if self.use_normalization:
          # See section 2.3 of the Soft MoE paper
          # Note that paper points out how needed with pre-norm (possibly not with post-norm?)
          self.normX =  L2NormScaled(1, False)
          self.normPhi = L2NormScaled(0, True)
       
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # NOTE: Copy weight initialization from 'nn.Linear.reset_parameters'
        # TODO: Check for initialization strategy from the paper
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the Soft-MoE layer, as described in:
            https://arxiv.org/pdf/2308.00951.pdf
        See: equations (1-3), algorithm 1, and figure 2

        einstein notation:
        - b: batch size
        - m: input sequence length
        - d: embedding dimension
        - n: num experts
        - p: num slots per expert
        - (n * p): total number of slots

        Args:
            x (Tensor): input tensor of shape (b, m, d)

        Returns:
            Tensor: output tensor of shape (b, m, d)
        """
#        if x.size(-1) != self.dim:
#            raise ValueError(
#                f"Expected x.size(-1)={x.size(-1)} to match embed_dim={self.dim}, "
#                f"but got {x.size(-1)}."
#            )
#        elif x.ndim != 3:
#            raise ValueError(f"Expected input to have 3 dimensions, but got {x.ndim}.")

        if self.use_normalization:
            xPrepared = self.normX(x)
            phiPrepared = self.normPhi(self.phi)
        else:
            xPrepared = x
            phiPrepared = self.phi   

        logits = einsum(xPrepared, phiPrepared, "b m d, d n p -> b m n p")

        dispatch_weights = logits.softmax(dim=1)  # denoted 'D' in the paper # NOTE by DJE: the paper/code incorrectly had dim=0
        # NOTE: The 'torch.softmax' function does not support multiple values for the
        # 'dim' argument (unlike jax), so we are forced to flatten the last two dimensions.
        # Then, we rearrange the Tensor into its original shape.
        combine_weights = rearrange(
            logits.flatten(start_dim=2).softmax(dim=-1),
            "b m (n p) -> b m n p",
            n=self.num_experts,
        )

        # NOTE: To save memory, I don't rename the intermediate tensors Y, Ys, Xs.
        # Instead, I just overwrite the 'x' variable.  The names from the paper are
        # included in a comment for each line below.
        x = einsum(x, dispatch_weights, "b m d, b m n p -> b n p d")  # Xs

        if (not self.only_second_layer):        
          x = self.experts1(x) # First linear layer
          x = nn.functional.relu(x).square() # Squared RELU nonlinearity
        
        x = self.experts2(x)  # Ys
        x = einsum(x, combine_weights, "b n p d, b m n p -> b m d")  # Y

        return x


    def extra_repr(self) -> str:
        return (
            f"in_features={self.dim}, out_features={self.ffn_dim}, "
            f"num_experts={self.num_experts}, slots_per_expert={self.slots_per_expert}, "
            f"bias={self.bias}"
        )

