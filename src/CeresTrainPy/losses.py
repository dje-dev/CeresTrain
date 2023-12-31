# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

# NOTE: this code derived from: https://github.com/Rocketknight1/minimal_lczero.

import torch
from torch import nn
from torch.nn import functional as F

class LossCalculator():
  """Class to compute and keep track of losses on various training target heads.
   """

  def __init__(self):
    super().__init__()

    self.MASK_POLICY_VALUE = -1E5 #; // for illegal moves

    # Keep running statistics (counts/totals) in between calls to reset_counters.
    self.PENDING_COUNT = 0
    self.PENDING_VALUE_LOSS = 0
    self.PENDING_POLICY_LOSS = 0
    self.PENDING_VALUE_ACC = 0
    self.PENDING_POLICY_ACC = 0
    self.PENDING_MLH_LOSS = 0
    self.PENDING_UNC_LOSS = 0

  def reset_counters(self):
    self.PENDING_COUNT = 0
    self.PENDING_VALUE_LOSS = 0
    self.PENDING_POLICY_LOSS = 0
    self.PENDING_VALUE_ACC = 0
    self.PENDING_POLICY_ACC = 0
    self.PENDING_MLH_LOSS = 0
    self.PENDING_UNC_LOSS = 0

  @property
  def LAST_VALUE_LOSS(self):
    return self.PENDING_VALUE_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_POLICY_LOSS(self):
    return self.PENDING_POLICY_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_VALUE_ACC(self):
    return self.PENDING_VALUE_ACC / self.PENDING_COUNT
  
  @property
  def LAST_POLICY_ACC(self):
    return self.PENDING_POLICY_ACC / self.PENDING_COUNT

  @property
  def LAST_MLH_LOSS(self):
    return self.PENDING_MLH_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_UNC_LOSS(self):
    return self.PENDING_UNC_LOSS / self.PENDING_COUNT
  
  
  def calc_accuracy(self, target: torch.Tensor, output: torch.Tensor, apply_masking : bool) -> float:
    if apply_masking:
      legalMoves = target.greater(0)
      illegalMaskValue = torch.zeros_like(output).add_(self.MASK_POLICY_VALUE)
      output = torch.where(legalMoves, output, illegalMaskValue)
    
    max_scores, max_idx_class = target.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    max_scores_out, max_idx_class_out = output.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
    n = target.size(0)
    acc = (max_idx_class == max_idx_class_out).sum().item() / n
    return 100 * acc


  def policy_loss(self, target: torch.Tensor, output: torch.Tensor):
    legalMoves = target.greater(0)
    illegalMaskValue = torch.zeros_like(output).add_(self.MASK_POLICY_VALUE)
    output = torch.where(legalMoves, output, illegalMaskValue)

    loss = nn.CrossEntropyLoss().forward(output, target)

    self.PENDING_POLICY_LOSS += loss.item()
    self.PENDING_POLICY_ACC += self.calc_accuracy(target, output, True)
    self.PENDING_COUNT = self.PENDING_COUNT + 1 # increment only for policy, not other losses

#   cos = nn.CosineSimilarity(dim=1, eps=1e-6) # cosine similarity and correlation metrics are related
#   pearson = cos(target - target.mean(dim=1, keepdim=True), output - output.mean(dim=1, keepdim=True))
#   print ('policy ', loss.item(), ' ', (sum(pearson) / len(pearson)).item(), '  acc ', self.LAST_POLICY_ACC)
#   return 100 * torch.nn.functional.mse_loss(output, target)

    return loss


  def value_loss(self, target: torch.Tensor, output: torch.Tensor):
    loss = nn.CrossEntropyLoss().forward(output, target)

    self.PENDING_VALUE_LOSS += loss.item()
    self.PENDING_VALUE_ACC += self.calc_accuracy(target, output, False)

    return loss


  def moves_left_loss(self, target: torch.Tensor, output: torch.Tensor):
    # Scale the loss to similar range as other losses.
    self.POST_SCALE = 5.0
    loss = self.POST_SCALE * F.huber_loss(output, target, reduction="mean", delta=0.5)

    self.PENDING_MLH_LOSS += loss.item()

    return loss


  def unc_loss(self, target: torch.Tensor, output: torch.Tensor):
    # Scale the loss to similar range as other losses.
    self.POST_SCALE = 150.0
    loss = self.POST_SCALE * F.huber_loss(output, target, reduction="mean", delta=0.5)

    self.PENDING_UNC_LOSS += loss.item()

    return loss
