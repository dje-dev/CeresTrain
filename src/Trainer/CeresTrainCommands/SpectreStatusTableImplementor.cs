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
using System.Collections.Generic;
using Spectre.Console;

#endregion

namespace CeresTrain.Trainer
{
  public partial class TrainingStatusTable
  {
    /// <summary>
    /// Implementation for the Spectre.Console library.
    /// </summary>
    public class SpectreStatusTableImplementor : TrainingStatusTableImplementor
    {
      /// <summary>
      /// Sequence of all emitted training status records.
      /// </summary>
      public readonly List<TrainingStatusRecord> TrainingStatusRecords = new();

      TrainingStatusRecord currentRecord;

      LiveDisplayContext context;
      Table table = new Table();

      string title;

      float lastTotalLoss = 0;


      /// <summary>
      /// Constructor.
      /// </summary>
      /// <param name="maxPositions"></param>
      public SpectreStatusTableImplementor(string title, long maxPositions) : base(maxPositions)
      {
        table = new Table();
        table.Title(title);
        table.AddColumn("Time");
        table.AddColumn(new TableColumn("Secs").RightAligned());
        table.AddColumn(new TableColumn("Positions").RightAligned());
        table.AddColumn(new TableColumn("Pos/sec").RightAligned());
        table.AddColumn(new TableColumn("Diff").RightAligned());
        table.AddColumn(new TableColumn("Loss").RightAligned());
        table.AddColumn(new TableColumn("PolicyLoss").RightAligned());
        table.AddColumn(new TableColumn("ValueLoss").RightAligned());
        table.AddColumn(new TableColumn("PolicyAcc").RightAligned());
        table.AddColumn(new TableColumn("ValueAcc").RightAligned());
        table.AddColumn(new TableColumn("LR").RightAligned());
      }

      public override void RunTraining(Action trainingLoop)
      {
        AnsiConsole.Live(table).Start(ctx =>
        {
          context = ctx;
          trainingLoop();

          // Add final row.
          TrainingStatusRecords.Add(currentRecord);
        });
      }

      public override void SetTitle(string title)
      {
        this.title = title;
      }

      public override void UpdateInfo(int numRowsAdded, bool endRow, 
                                     float posPerSecond, DateTime time, float elapsedSecs, long numPositions, 
                                     float totalLoss, float valueLoss, float valueAcc, float policyLoss, float policyAcc, float curLR)
      {
        currentRecord = new TrainingStatusRecord(time, elapsedSecs, posPerSecond, numPositions,
                                                 totalLoss, valueLoss, valueAcc,
                                                 policyLoss, policyAcc, curLR);

        int curRowNum = numRowsAdded - 1;

        if (numRowsAdded > 0)
        {
          table.UpdateCell(curRowNum, 0, $"{time.ToString("HH\\:mm\\:ss")}");
          table.UpdateCell(curRowNum, 1, $"{elapsedSecs:F1}");
          table.UpdateCell(curRowNum, 2, $"{numPositions:N0}");
          table.UpdateCell(curRowNum, 3, $"{posPerSecond:N0}");

          if (lastTotalLoss != 0)
          {
            float deltaLoss = totalLoss - lastTotalLoss;
            string colorStr = deltaLoss > 0 ? "red" : "green";
            table.UpdateCell(curRowNum, 4, $"[{colorStr}]{deltaLoss:F3}[/]");
          }
          table.UpdateCell(curRowNum, 5, $"{totalLoss:F3}");

          table.UpdateCell(curRowNum, 6, $"{policyLoss:F3}");
          table.UpdateCell(curRowNum, 7, $"{valueLoss:F3}");
          table.UpdateCell(curRowNum, 8, $"{100 * policyAcc:F2}%");
          table.UpdateCell(curRowNum, 9, $"{100 * valueAcc:F2}%");
          table.UpdateCell(curRowNum, 10, $"{Math.Round(curLR, 6):F6}");
          context.Refresh();
        }

        if (endRow)
        {
          lastTotalLoss = totalLoss;
          table.AddRow("", "0", "", "0", "0");
          TrainingStatusRecords.Add(currentRecord);
        }
      }
    }

  }
}
