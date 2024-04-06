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

    self.MASK_POLICY_VALUE = -1E5 # for illegal moves

    # Keep running statistics (counts/totals) in between calls to reset_counters.
    self.reset_counters()
    self.ce_loss = nn.CrossEntropyLoss()
    
  def reset_counters(self):
    self.PENDING_COUNT = 0
    self.PENDING_VALUE_LOSS = 0
    self.PENDING_POLICY_LOSS = 0
    self.PENDING_VALUE_ACC = 0
    self.PENDING_POLICY_ACC = 0
    self.PENDING_MLH_LOSS = 0
    self.PENDING_UNC_LOSS = 0
    self.PENDING_VALUE2_LOSS = 0
    self.PENDING_Q_DEVIATION_LOWER_LOSS = 0
    self.PENDING_Q_DEVIATION_UPPER_LOSS = 0
    self.PENDING_VALUE_DIFF_LOSS = 0
    self.PENDING_VALUE2_DIFF_LOSS = 0
    self.PENDING_ACTION_LOSS = 0

  @property
  def LAST_VALUE_LOSS(self):
    return self.PENDING_VALUE_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_VALUE2_LOSS(self):
    return self.PENDING_VALUE2_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_VALUE_DIFF_LOSS(self):
    return self.PENDING_VALUE_DIFF_LOSS / self.PENDING_COUNT
  
  @property
  def LAST_VALUE2_DIFF_LOSS(self):
    return self.PENDING_VALUE2_DIFF_LOSS / self.PENDING_COUNT

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

  @property
  def LAST_Q_DEVIATION_LOWER_LOSS(self):
    return self.PENDING_Q_DEVIATION_LOWER_LOSS / self.PENDING_COUNT

  @property
  def LAST_Q_DEVIATION_UPPER_LOSS(self):
    return self.PENDING_Q_DEVIATION_UPPER_LOSS / self.PENDING_COUNT

  @property
  def LAST_ACTION_LOSS(self):
    return self.PENDING_ACTION_LOSS / self.PENDING_COUNT
  
  
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


  def entropy(self, probabilities : torch.Tensor):
    # entropy is same as cross entropy with itself
    clipped_probabilities = torch.clamp(probabilities + 1e-6, min=1e-6)
    return torch.nn.functional.cross_entropy(torch.log(clipped_probabilities),clipped_probabilities)

  def policy_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool):
    legalMoves = target.greater(0)
    illegalMaskValue = torch.zeros_like(output).add_(self.MASK_POLICY_VALUE)
    output = torch.where(legalMoves, output, illegalMaskValue)

    entropy = self.entropy(target) if subtract_entropy else 0.0
    loss = self.ce_loss.forward(output, target) - entropy
    
    self.PENDING_POLICY_LOSS += loss.item()
    self.PENDING_POLICY_ACC += self.calc_accuracy(target, output, True)
    self.PENDING_COUNT = self.PENDING_COUNT + 1 # increment only for policy, not other losses

#   cos = nn.CosineSimilarity(dim=1, eps=1e-6) # cosine similarity and correlation metrics are related
#   pearson = cos(target - target.mean(dim=1, keepdim=True), output - output.mean(dim=1, keepdim=True))
#   print ('policy ', loss.item(), ' ', (sum(pearson) / len(pearson)).item(), '  acc ', self.LAST_POLICY_ACC)
#   return 100 * torch.nn.functional.mse_loss(output, target)

    return loss


  def value_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool):
    entropy = self.entropy(target) if subtract_entropy else 0.0
    loss = self.ce_loss.forward(output, target) - entropy
    self.PENDING_VALUE_LOSS += loss.item()
    self.PENDING_VALUE_ACC += self.calc_accuracy(target, output, False)
    return loss

  def value2_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool):
    entropy = self.entropy(target) if subtract_entropy else 0.0
    loss = self.ce_loss.forward(output, target) - entropy
    self.PENDING_VALUE2_LOSS += loss.item()
    return loss

  def value_diff_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool):
    entropy = self.entropy(target) if subtract_entropy else 0.0
    loss = self.ce_loss.forward(output, target) - entropy
    
    self.PENDING_VALUE_DIFF_LOSS += loss.item()
    return loss

  def value2_diff_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool):
    entropy = self.entropy(target) if subtract_entropy else 0.0
    loss = self.ce_loss.forward(output, target) - entropy
   
    self.PENDING_VALUE2_DIFF_LOSS += loss.item()
    return loss

  def action_loss(self, target: torch.Tensor, output: torch.Tensor, subtract_entropy : bool):
    entropy = self.entropy(target) if subtract_entropy else 0.0
    loss = self.ce_loss(output, F.softmax(target, dim=-1)) - entropy
  
    self.PENDING_ACTION_LOSS += loss.item()
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


  def q_deviation_lower_loss(self, target: torch.Tensor, output: torch.Tensor):
    self.POST_SCALE = 10.0
    loss = self.POST_SCALE * nn.MSELoss().forward(output, target)
    self.PENDING_Q_DEVIATION_LOWER_LOSS += loss.item()
    return loss


  def q_deviation_upper_loss(self, target: torch.Tensor, output: torch.Tensor):
    self.POST_SCALE = 10.0
    loss = self.POST_SCALE * nn.MSELoss().forward(output, target)
    self.PENDING_Q_DEVIATION_UPPER_LOSS += loss.item()
    return loss
