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
using System.Collections.Generic;

using Spectre.Console;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Subclass of TrainingStatusTableImplementor which outputs to the console,
  /// suitable for concurrent use by multiple threads.
  /// </summary>
  public class BatchTrainingStatusTable : TrainingStatusTableImplementor
  {
    LiveDisplayContext context;
    Table table = new Table();

    public BatchTrainingStatusTable(long maxPositions) : base(maxPositions)
    {
      table = new Table();
      table.Title("BATCH");

      table.AddColumn(new TableColumn("Time"));
      table.AddColumn(new TableColumn("Secs").RightAligned());
      table.AddColumn(new TableColumn("Config"));
      table.AddColumn(new TableColumn("Host"));

      table.AddColumn(new TableColumn("Positions").RightAligned());
      table.AddColumn(new TableColumn("Pos/sec").RightAligned());
      table.AddColumn(new TableColumn("Loss").RightAligned());
      table.AddColumn(new TableColumn("PolLoss").RightAligned());
      table.AddColumn(new TableColumn("ValLoss").RightAligned());
      table.AddColumn(new TableColumn("Val2Loss").RightAligned());
      table.AddColumn(new TableColumn("MLHLoss").RightAligned());
      table.AddColumn(new TableColumn("UNCLoss").RightAligned());
      table.AddColumn(new TableColumn("PolAcc").RightAligned());
      table.AddColumn(new TableColumn("ValAcc").RightAligned());
      table.AddColumn(new TableColumn("QDev").RightAligned());
      table.AddColumn(new TableColumn("PolUnc").RightAligned());
      table.AddColumn(new TableColumn("ValD").RightAligned());
      table.AddColumn(new TableColumn("Val2D").RightAligned());
      table.AddColumn(new TableColumn("ActLoss").RightAligned());
      table.AddColumn(new TableColumn("ActUncLoss").RightAligned());
      table.AddColumn(new TableColumn("LR").RightAligned());
    }

    public override void RunTraining(Action processor)
    {
      lastTime = DateTime.Now;
      startTime = DateTime.Now;
      //var live = AnsiConsole.Live(table);
      //live.Start(ctx => context = ctx);
      processor();
    }

    public override void SetTitle(string title)
    {
    
    }

    Dictionary<string, string[]> Lasts = new();

    static readonly object lockObj = new object();
    DateTime lastTime;
    DateTime startTime;

    Dictionary<string, (DateTime, long)> priorLastRows = new();

    public override void UpdateInfo(string configID, string host, int numRowsAdded, bool endRow, 
                                    float posPerSecond, DateTime time, float elapsedSecs, long numPositions, 
                                    float totalLoss, float valueLoss, float valueAcc, 
                                    float policyLoss, float policyAcc,
                                    float mlhLoss, float uncLoss,
                                    float value2Loss, float qDeviationMaxLoss, float policyUncertaintyLoss,
                                    float valueDLoss, float value2DLoss,
                                    float actionLoss, float actionUncertaintyLoss,
                                    float curLR)
    {

      lock(lockObj)
      {
        // Compute posPerSecond which is specific for this config.
        (DateTime, long) thisPriorLast;
        posPerSecond = 0;
        if (priorLastRows.TryGetValue(configID, out thisPriorLast))
        {
          float elapsedTime = (float)(DateTime.Now - thisPriorLast.Item1).TotalSeconds;
          long elapsedRows = numPositions - thisPriorLast.Item2;
          posPerSecond = elapsedRows / elapsedTime;
        }
        priorLastRows[configID] = (DateTime.Now, numPositions);

        Lasts[configID] = [configID, host,
                           numPositions.ToString(),
                           MathF.Round(posPerSecond, 0).ToString(), MathF.Round(totalLoss, 3).ToString(),

                           MathF.Round(policyLoss, 3).ToString(), MathF.Round(valueLoss, 3).ToString(),
                           MathF.Round(value2Loss, 3).ToString(),
                           MathF.Round(mlhLoss, 3).ToString(), MathF.Round(uncLoss, 3).ToString(),
                           MathF.Round(100*policyAcc, 2).ToString(), MathF.Round(100*valueAcc, 2).ToString(),
                           MathF.Round(qDeviationMaxLoss, 3).ToString(), MathF.Round(policyUncertaintyLoss, 3).ToString(),
                           MathF.Round(valueDLoss, 3).ToString(), MathF.Round(value2DLoss, 3).ToString(),
                           MathF.Round(actionLoss, 3).ToString(),MathF.Round(actionUncertaintyLoss, 3).ToString(),

                           curLR.ToString()];

        int UPDATE_INTERVAL_SECS = numPositions > 10_000_000 ? 180 : 30;
        if ((DateTime.Now - lastTime).TotalSeconds > UPDATE_INTERVAL_SECS)
        {
          float tableElapsedSecs = (float)(DateTime.Now - startTime).TotalSeconds;
          while (table.Rows.Count > 0)
          {
            table.RemoveRow(0);
          }
          foreach (KeyValuePair<string, string[]> configs in Lasts)
          {
            List<string> columns =
            [
              DateTime.Now.ToString("HH\\:mm\\:ss"),
              Math.Round(tableElapsedSecs, 0).ToString(),
              .. configs.Value,
            ];
            table.AddRow(columns.ToArray());
          }
          AnsiConsole.Write(table);
          lastTime = DateTime.Now;
        }

        //this.context.Refresh();
      } 
    }
  }
}