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
using System.Text.Json.Serialization;
using TorchSharp.Modules;

#endregion

namespace CeresTrain.Trainer
{
  public readonly record struct TrainingLossSummary
  {
    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float TotalLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float ValueLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float ValueAccuracy { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float PolicyLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float PolicyAccuracy { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float MLHLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float UNCLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float Value2Loss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float QDeviationLowerLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float QDeviationUpperLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float PolicyUncertaintyLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float ValueDLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float Value2DLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float ActionLoss { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float ActionUncertaintyLoss { get; init; }

    public int NumFineTuneAnchorPositions { get; init; }

    public int NumFineTuneNonAnchorPositions { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float AvgFineTuneAnchorError { get; init; }

    [JsonNumberHandling(JsonNumberHandling.AllowNamedFloatingPointLiterals)]
    public float AvgFineTuneNonAnchorError { get; init; }


    public TrainingLossSummary(float totalLoss,
                               float valueLoss, float valueAccuracy,
                               float policyLoss, float policyAccuracy,
                               float mlhLoss, float uncLoss,
                               float value2Loss,
                               float qDeviationLowerLoss, float qDeviationUpperLoss,
                               float policyUncertaintyLoss,
                               float valueDLoss, float value2DLoss,
                               float actionLoss, float actionUncertaintyLoss,
                               int numFineTuneAnchorPositions, int numFineTuneNonAnchorPositions,
                               float avgFineTuneAnchorError, float avgFineTuneNonAnchorError)
    {
      TotalLoss = totalLoss;
      ValueLoss = valueLoss;
      ValueAccuracy = valueAccuracy;
      PolicyLoss = policyLoss;
      PolicyAccuracy = policyAccuracy;
      MLHLoss = mlhLoss;
      UNCLoss = uncLoss;
      Value2Loss = value2Loss;
      QDeviationLowerLoss = qDeviationLowerLoss;
      QDeviationUpperLoss = qDeviationUpperLoss;
      PolicyUncertaintyLoss = policyUncertaintyLoss;
      ValueDLoss = valueDLoss;
      Value2DLoss = value2DLoss;
      ActionLoss = actionLoss;
      ActionUncertaintyLoss = actionUncertaintyLoss;
      NumFineTuneAnchorPositions = numFineTuneAnchorPositions;
      NumFineTuneNonAnchorPositions = numFineTuneNonAnchorPositions;
      AvgFineTuneAnchorError = avgFineTuneAnchorError;
      AvgFineTuneNonAnchorError = avgFineTuneNonAnchorError;
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
