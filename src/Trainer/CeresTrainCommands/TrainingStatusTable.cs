﻿#region License notice

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
using System.Reflection;
using System.Collections.Generic;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Manages the GUI (a table on the Console) that outputs 
  /// with live updates the status on the progress of training.
  /// </summary>
  public partial class TrainingStatusTable
  {
    public readonly record struct TrainingStatusRecord(DateTime Time, float ElapsedSecs,
                                                     float PosPerSecond, long NumPositions,
                                                     float TotalLoss,
                                                     float ValueLoss, float ValueAcc,
                                                     float PolicyLoss, float PolicyAcc,
                                                     float CurLR);

    const float INTERVAL_NEW_ROW_FIRST_HOUR = 60;       // New row very minute for first hour
    const float INTERVAL_NEW_ROW_LATER_HOURS = 10 * 60; // New row every 10 minutes after first hour


    /// <summary>
    /// Sequence of all emitted training status records.
    /// </summary>
    public readonly List<TrainingStatusRecord> TrainingStatusRecords = new();

    TrainingStatusTableImplementor implementor;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="title"></param>
    /// <param name="maxPositions"></param>
    public TrainingStatusTable(string id, string title, long maxPositions)
    {
      // Linqpad needs a custom table implementor because the AnsiConsole
      // assumed by SpectreConsole is not available.
      bool runningUnderLinqPad = Assembly.GetEntryAssembly().ToString().ToUpper().StartsWith("LINQPAD");
      implementor = runningUnderLinqPad ? new LINQPadStatusTableImplementor(maxPositions)
                                        : new SpectreStatusTableImplementor(title, maxPositions);

      implementor.SetTitle(title);
    }


    int numRowsAdded = 0;
    DateTime lastRowAdded = DateTime.Now;
    DateTime timeStart = DateTime.Now;
    float lastTotalLoss = 0;
    long lastRowNumPositions = 0;
    float intervalBetweenRows = INTERVAL_NEW_ROW_FIRST_HOUR;

    TrainingStatusRecord currentRecord;


    /// <summary>
    /// Posts an update to the table with new training statistics.
    /// </summary>
    /// <param name="time"></param>
    /// <param name="elapsedSecs"></param>
    /// <param name="numPositions"></param>
    /// <param name="totalLoss"></param>
    /// <param name="valueLoss"></param>
    /// <param name="valueAcc"></param>
    /// <param name="policyLoss"></param>
    /// <param name="policyAcc"></param>
    /// <param name="curLR"></param>
    public void UpdateInfo(DateTime time, float elapsedSecs, long numPositions,
                           float totalLoss, float valueLoss, float valueAcc,
                           float policyLoss, float policyAcc, float curLR)
    {
      float timeSinceStart = (float)(DateTime.Now - timeStart).TotalSeconds;
      float timeSinceNewRow = (float)(DateTime.Now - lastRowAdded).TotalSeconds;
      float posPerSecond = (numPositions - lastRowNumPositions) / timeSinceNewRow;

      currentRecord = new TrainingStatusRecord(time, elapsedSecs, posPerSecond, numPositions,
                                         totalLoss, valueLoss, valueAcc,
                                         policyLoss, policyAcc, curLR);

      int curRowNum = numRowsAdded - 1;

      if (numRowsAdded > 0)
      {
        implementor.UpdateInfo(numRowsAdded, false, posPerSecond, time, elapsedSecs, numPositions, totalLoss,
                               valueLoss, valueAcc, policyLoss, policyAcc, curLR);
      }

      if (numRowsAdded == 0 || timeSinceNewRow > intervalBetweenRows)
      {
        // During first hour add new rows more often.
        if (intervalBetweenRows == INTERVAL_NEW_ROW_FIRST_HOUR && timeSinceStart > 3600)
        {
          intervalBetweenRows = INTERVAL_NEW_ROW_LATER_HOURS;
        }

        implementor.UpdateInfo(numRowsAdded, true, posPerSecond, time, elapsedSecs, numPositions, totalLoss,
                               valueLoss, valueAcc, policyLoss, policyAcc, curLR);
        TrainingStatusRecords.Add(currentRecord);

        lastTotalLoss = totalLoss;
        lastRowAdded = DateTime.Now;
        lastRowNumPositions = numPositions;
        numRowsAdded++;
      }

    }


    /// <summary>
    /// Runs the specified training loop code inside a live table update context.
    /// </summary>
    /// <param name="trainingLoop"></param>
    public void RunTraining(Action trainingLoop)
    {
      implementor.RunTraining(trainingLoop);

      // Add final row.
      TrainingStatusRecords.Add(currentRecord);
    }

  }
}
