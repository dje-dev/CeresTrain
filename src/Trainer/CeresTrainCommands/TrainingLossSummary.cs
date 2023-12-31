#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/cerestrain.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with CeresTrain. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Summary of loss values/statistics during training.
  /// </summary>
  /// <param name="TotalLoss"></param>
  /// <param name="ValueLoss"></param>
  /// <param name="ValueAccuracy"></param>
  /// <param name="PolicyLoss"></param>
  /// <param name="PolicyAccuracy"></param>
  [Serializable]
  public readonly record struct TrainingLossSummary(float TotalLoss,
                                                    float ValueLoss, float ValueAccuracy,
                                                    float PolicyLoss, float PolicyAccuracy)
  {
  }
}
