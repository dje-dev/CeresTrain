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

#endregion

namespace CeresTrain.TrainCommands
{
  /// <summary>
  /// Captures the information from a single TRAIN: line
  /// emitted by the Python training code to Console.
  /// </summary>
  internal readonly record struct CeresTrainProgressLoggingLine
  {
    /// <summary>
    /// Total number of positions trained on so far.
    /// </summary>
    public readonly long NumPos;

    /// <summary>
    /// Total loss (weighted appropriately).
    /// </summary>
    public readonly float TotalLoss;

    /// <summary>
    /// Value head loss (unweighted).
    /// </summary>
    public readonly float LastValueLoss;

    /// <summary>
    /// Policy head loss (unweighted).
    /// </summary>
    public readonly float LastPolicyLoss;

    /// <summary>
    /// Value head accuracy.
    /// </summary>
    public readonly float LastValueAcc;

    /// <summary>
    /// Policy head accuracy.
    /// </summary>
    public readonly float LastPolicyAcc;

    /// <summary>
    /// MLH (moves left head) head loss (unweighted).
    /// </summary>
    public readonly float LastMLHLoss;

    /// <summary>
    /// UNC (uncertainty) head loss (unweighted).
    /// </summary>
    public readonly float LastUNCLoss;

    /// <summary>
    /// Value head 2 (secondary) loss.
    /// </summary>
    public readonly float LastValue2Loss;

    /// <summary>
    /// Value head difference loss.
    /// </summary>
    public readonly float LastValueDLoss;

    /// <summary>
    /// Value head (secondary) difference loss.
    /// </summary>
    public readonly float LastValue2DLoss;

    /// <summary>
    /// Action head loss.
    /// </summary>
    public readonly float LastActionLoss;

    /// <summary>
    /// Action uncertainty head loss.
    /// </summary>
    public readonly float LastActionUncertaintyLoss;

    /// <summary>
    /// Foward Q deviation lower loss.
    /// </summary>
    public readonly float LastQDeviationLowerLoss;

    /// <summary>
    /// Foward Q deviation upper loss.
    /// </summary>
    public readonly float LastQDeviationUpperLoss;

    /// <summary>
    /// Policy Uncertainty loss.
    /// </summary>
    public readonly float LastPolicyUncertaintyLoss;

    /// <summary>
    /// Learning rate
    /// </summary>
    public readonly float LastLR;


    /// <summary>
    /// Constructor from a single line of the training progress logging.
    /// </summary>
    /// <param name="input"></param>
    public CeresTrainProgressLoggingLine(string input)
    {
      string data = input.Replace("TRAIN: ", "").Trim();
      string[] parts = data.Split(',');

      NumPos = long.Parse(parts[0].Trim());
      TotalLoss = float.Parse(parts[1].Trim());
      LastValueLoss = float.Parse(parts[2].Trim());
      LastPolicyLoss = float.Parse(parts[3].Trim());
      LastValueAcc = float.Parse(parts[4].Trim());
      LastPolicyAcc = float.Parse(parts[5].Trim());

      LastMLHLoss = float.Parse(parts[6].Trim());
      LastUNCLoss = float.Parse(parts[7].Trim());
      LastValue2Loss = float.Parse(parts[8].Trim());
      LastQDeviationLowerLoss = float.Parse(parts[9].Trim());
      LastQDeviationUpperLoss = float.Parse(parts[10].Trim());
      LastPolicyUncertaintyLoss = float.Parse(parts[11].Trim());

      LastValueDLoss = float.Parse(parts[12].Trim());
      LastValue2DLoss = float.Parse(parts[13].Trim());
      LastActionLoss = float.Parse(parts[14].Trim());
      LastActionUncertaintyLoss = float.Parse(parts[15].Trim());
      LastLR = float.Parse(parts[16].Trim());
    }
  }
}
