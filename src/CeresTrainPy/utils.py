# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

import os
import sys
import torch


"""
Logs the FLOPS (floating point operations) of the model (per position)
during either the inference (forward) or training (forward + backward) passes.

Note that if custom layers (such as TransformerEngine) are in use,
these FLOPS counts may be incorrect (underestimated).
"""
def calc_flops(model_test, batch, loss_calc, optimizer, num_pos, batch_size, calc_backward=False):
  from torch.utils.flop_counter import FlopCounterMode
  flop_counter = FlopCounterMode(display=False)
    
  with flop_counter:
    policy_out, value_out, moves_left_out, unc_out, value2_out, q_deviation_lower, q_deviation_upper, uncertainty_policy_out, _, _, _ = model_test(batch['squares'], None)       
    loss = model_test.compute_loss(loss_calc, batch, policy_out, value_out, moves_left_out, unc_out,
                                        value2_out, q_deviation_lower, q_deviation_upper, uncertainty_policy_out, 
                                        None, None, 
                                        None, None,
                                        None,
                                        0, num_pos, 0, False)
        
    if calc_backward:
      loss.backward()
    
  flops_per_pos = flop_counter.get_total_flops() / batch_size
  print(f"INFO: MODEL_GFLOPS_" + ("TRAIN:" if calc_backward else "INFERENCE:"), round(flops_per_pos / 1_000_000_000, 3))
