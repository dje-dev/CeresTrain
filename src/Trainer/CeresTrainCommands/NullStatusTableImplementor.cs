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
  /// Implementation for a null status table, which does not output anything.
  /// </summary>
  public class NullStatusTableImplementor : TrainingStatusTableImplementor
  {
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="maxPositions"></param>
    public NullStatusTableImplementor(long maxPositions) : base(maxPositions)
    {
    }

    public override void RunTraining(Action processor) => processor();

    public override void SetTitle(string title)
    {      
    }

    public override void UpdateInfo(string configID, string host, int numRowsAdded, bool endRow, float posPerSecond, DateTime time, float elapsedSecs, long numPositions, float totalLoss, float valueLoss, float valueAcc, float policyLoss, float policyAcc, float mlhLoss, float uncLoss, float value2Loss, float qDeviationLowerLoss, float qDeviationUpperLoss, float policyUncertaintyLoss, float valueDLoss, float value2DLoss, float actionLoss, float actionUncertaintyLoss, float curLR)
    {      
    }
  }

}
