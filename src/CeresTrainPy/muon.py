import os
import torch
import torch.distributed as dist
from torch import Tensor

# Copied from https://github.com/KellerJordan/Muon

# NOTE: CeresTrain integration not working, seemingly due to incompatability with Fabric
#       (error about process group not initialized)
# NOTE: See bottom of this file for approximate changes needed to train.py

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, fabric, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5):
        self.fabric = fabric
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        world_size = fabric.world_size
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=b,
                update_buffer_views=[b[i] for i in range(world_size)]
            )
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
      for group in self.param_groups:
        update_buffer: Tensor = group["update_buffer"]
        update_buffer_views: list[Tensor] = group["update_buffer_views"]
        params: list[Tensor] = group["params"]
        
        for base_i in range(0, len(params), self.fabric.world_size):
          if base_i + self.fabric.global_rank < len(params):
              p = params[base_i + self.fabric.global_rank]
              g = p.grad
              assert g is not None
              state = self.state[p]
              if "momentum_buffer" not in state:
                  state["momentum_buffer"] = torch.zeros_like(g)
              buf: Tensor = state["momentum_buffer"]
              buf.lerp_(g, 1 - group["momentum"])
              g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
              if g.ndim == 4:
                  g = g.view(len(g), -1)
              g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
          else:
              g = torch.zeros_like(update_buffer_views[self.fabric.global_rank])

          # Fabric synchronous gather:
          #self.fabric.all_gather_into_tensor(update_buffer, g)
          gathered = self.fabric.all_gather(g)

          params_world = params[base_i : base_i + self.fabric.world_size]
          for p_world, g_world in zip(params_world, update_buffer_views):
              p_world.mul_(1 - group["lr"] * group["weight_decay"])
              p_world.add_(g_world.view_as(p_world),
                            alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)


'''
+from muon import Muon


+world_size = len(devices)
+rank = 0 if world_size == 1 else dist.get_rank()


+  # Find all parameters which are 2D or higher -- these should be optimized by Muon
+  muon_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
+  # Find everything else -- these should be optimized by AdamW
+  adamw_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
+  # Create the optimizer
+  #optimizers = [Muon(muon_params, lr=0.02, momentum=0.95),
+  #              torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)]
+
+  elif config.Opt_Optimizer == 'Muon':
+    optimizer = Muon(muon_params, lr=LR, weight_decay=WEIGHT_DECAY, momentum=config.Opt_Beta1, nesterov=True, ns_steps=5, rank=rank, world_size=world_size)
+# RECOMMENDED:   params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
+

-+world_size = len(devices)
-rank = 0 if world_size == 1 else dist.get_rank()

'''