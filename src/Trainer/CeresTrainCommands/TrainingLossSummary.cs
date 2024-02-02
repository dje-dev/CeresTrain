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
  /// <param name="MLHLoss"></param>
  /// <param name="UNCLoss"></param>
  /// <param name="Value2Loss"></param>
  /// <param name="QDeviationLowerLoss"></param>
  /// <param name="QDeviationUpperLoss"></param>

  [Serializable]
  public readonly record struct TrainingLossSummary(float TotalLoss,
                                                    float ValueLoss, float ValueAccuracy,
                                                    float PolicyLoss, float PolicyAccuracy,
                                                    float MLHLoss, float UNCLoss,
                                                    float Value2Loss,
                                                    float QDeviationLowerLoss, float QDeviationUpperLoss)
  {
    /// <summary>
    /// Returns a string summarizing the loss values.
    /// </summary>
    public string SummaryStr
    {
      get
      {
        return $"TL:{TotalLoss,7:F3}  " +
               $"VL:{ValueLoss,7:F3}  " +
               $"VA:{ValueAccuracy,7:F3}  " +
               $"PL:{PolicyLoss,7:F3}  " +
               $"PA:{PolicyAccuracy,7:F3}  " +
               $"MLH:{MLHLoss,7:F3}  " +
               $"UNC:{UNCLoss,7:F3}  " +
               $"V2L:{Value2Loss,7:F3}  " +
               $"QDLL:{QDeviationLowerLoss,7:F3}  " +
               $"QDUL:{QDeviationUpperLoss,7:F3}";
      }
    }
  }

}

