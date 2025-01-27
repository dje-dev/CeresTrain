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
using TorchSharp.Modules;

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
  /// <param name="PolicyUncertaintyLoss"></param>
  /// <param name="ValueDLoss"></param>
  /// <param name="Value2DLoss"></param>
  /// <param name="ActionLoss"></param>
  /// <param name="ActionUncertaintyLoss"></param>
  /// <param name="AvgFineTuneAnchorError"></param>
  /// <param name="AvgFineTuneNonAnchorError"></param>
  [Serializable]
  public readonly record struct TrainingLossSummary(float TotalLoss,
                                                    float ValueLoss, float ValueAccuracy,
                                                    float PolicyLoss, float PolicyAccuracy,
                                                    float MLHLoss, float UNCLoss,
                                                    float Value2Loss,
                                                    float QDeviationLowerLoss, float QDeviationUpperLoss,
                                                    float PolicyUncertaintyLoss,
                                                    float ValueDLoss, float Value2DLoss,
                                                    float ActionLoss, float ActionUncertaintyLoss,
                                                    int NumFineTuneAnchorPositions, int NumFineTuneNonAnchorPositions,
                                                    float AvgFineTuneAnchorError, float AvgFineTuneNonAnchorError)
  {
    /// <summary>
    /// Returns a sanitized version of the loss summary where NaN values are replaced with -999
    /// (because writing JSON with NaN values is not allowed).  
    /// </summary>
    /// <returns></returns>
    public TrainingLossSummary ReplaceNaNWithMinus999()
    {
      return this with
      {
        TotalLoss = float.IsNaN(TotalLoss) ? -999 : TotalLoss,
        ValueLoss = float.IsNaN(ValueLoss) ? -999 : ValueLoss,
        ValueAccuracy = float.IsNaN(ValueAccuracy) ? -999 : ValueAccuracy,
        PolicyLoss = float.IsNaN(PolicyLoss) ? -999 : PolicyLoss,
        PolicyAccuracy = float.IsNaN(PolicyAccuracy) ? -999 : PolicyAccuracy,
        MLHLoss = float.IsNaN(MLHLoss) ? -999 : MLHLoss,
        UNCLoss = float.IsNaN(UNCLoss) ? -999 : UNCLoss,
        Value2Loss = float.IsNaN(Value2Loss) ? -999 : Value2Loss,
        QDeviationLowerLoss = float.IsNaN(QDeviationLowerLoss) ? -999 : QDeviationLowerLoss,
        QDeviationUpperLoss = float.IsNaN(QDeviationUpperLoss) ? -999 : QDeviationUpperLoss,
        PolicyUncertaintyLoss = float.IsNaN(PolicyUncertaintyLoss) ? -999 : PolicyUncertaintyLoss,
        ValueDLoss = float.IsNaN(ValueDLoss) ? -999 : ValueDLoss,
        Value2DLoss = float.IsNaN(Value2DLoss) ? -999 : Value2DLoss,
        ActionLoss = float.IsNaN(ActionLoss) ? -999 : ActionLoss,
        ActionUncertaintyLoss = float.IsNaN(ActionUncertaintyLoss) ? -999 : ActionUncertaintyLoss,
      };
    }



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
               $"QDEVL:{QDeviationLowerLoss,7:F3}  " +
               $"QDEVU:{QDeviationUpperLoss,7:F3}  " +
               $"UNCP:{PolicyUncertaintyLoss,7:F3}  " +
               $"VDL:{ValueDLoss,7:F3}  " +
               $"V2DL:{Value2DLoss,7:F3}" +
               $"ACT:{ActionLoss,7:F3}" +
               $"ACTU:{ActionUncertaintyLoss,7:F3}";
      }
    }
  }

}
