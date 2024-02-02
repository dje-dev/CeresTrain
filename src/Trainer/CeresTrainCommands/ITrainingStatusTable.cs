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
  /// Abstract base class providing primitives output methods for reporting training status in a table.
  /// </summary>
  public abstract class TrainingStatusTableImplementor
  {
    /// <summary>
    /// Maximum number of position expected for the training session.
    /// </summary>
    public readonly long MaxPositions;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="maxPositions"></param>
    public TrainingStatusTableImplementor(long maxPositions)
    {
      MaxPositions = maxPositions;
    } 


    /// <summary>
    /// Encapsulates running of code within specified Action. 
    /// </summary>
    /// <param name="processor"></param>
    public abstract void RunTraining(Action processor);


    /// <summary>
    /// Adds/updates a new line to the table.
    /// </summary>
    /// <param name="configID"></param>
    /// <param name="numRowsAdded"></param>
    /// <param name="endRow"></param>
    /// <param name="posPerSecond"></param>
    /// <param name="time"></param>
    /// <param name="elapsedSecs"></param>
    /// <param name="numPositions"></param>
    /// <param name="totalLoss"></param>
    /// <param name="valueLoss"></param>
    /// <param name="valueAcc"></param>
    /// <param name="policyLoss"></param>
    /// <param name="policyAcc"></param>
    /// <param name="mlhLoss"></param>
    /// <param name="uncLoss"></param>
    /// <param name="value2Loss"></param>
    /// <param name="qDeviationLowerLoss"></param>
    /// <param name="qDeviationUpperLoss"></param>
    /// <param name="curLR"></param>
    public abstract void UpdateInfo(string configID, int numRowsAdded, bool endRow, float posPerSecond, 
                                    DateTime time, float elapsedSecs, long numPositions, 
                                    float totalLoss, float valueLoss, float valueAcc, float policyLoss, float policyAcc,
                                    float mlhLoss, float uncLoss,
                                    float value2Loss, float qDeviationLowerLoss, float qDeviationUpperLoss,
                                    float curLR);


    /// <summary>
    /// Sets the title of the table.
    /// </summary>
    /// <param name="title"></param>
    public abstract void SetTitle(string title);
  }
}