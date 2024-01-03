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

using ManagedCuda.BasicTypes;
using Spectre.Console;
using System;
using System.Collections.Generic;

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

      table.AddColumn(new TableColumn("Positions").RightAligned());
      table.AddColumn(new TableColumn("Pos/sec").RightAligned());
      table.AddColumn(new TableColumn("Loss").RightAligned());
      table.AddColumn(new TableColumn("PolicyLoss").RightAligned());
      table.AddColumn(new TableColumn("ValueLoss").RightAligned());
      table.AddColumn(new TableColumn("PolicyAcc").RightAligned());
      table.AddColumn(new TableColumn("ValueAcc").RightAligned());
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

    public override void UpdateInfo(string configID, int numRowsAdded, bool endRow, float posPerSecond, DateTime time, float elapsedSecs, long numPositions, float totalLoss, float valueLoss, float valueAcc, float policyLoss, float policyAcc, float curLR)
    {

      lock(lockObj)
      {
        Lasts[configID] = [configID, 
                           numPositions.ToString(),
                           MathF.Round(posPerSecond, 0).ToString(), MathF.Round(totalLoss, 3).ToString(),
                           MathF.Round(policyLoss, 3).ToString(), MathF.Round(valueLoss, 3).ToString(),
                           MathF.Round(100*policyAcc, 2).ToString(), MathF.Round(100*valueAcc, 2).ToString(),
                           curLR.ToString()];

        const int UPDATE_INTERVAL_SECS = 20;
        if ((DateTime.Now - lastTime).TotalSeconds > UPDATE_INTERVAL_SECS)
        {
          float tableElapsedSecs = (float)(DateTime.Now - startTime).TotalSeconds;
          while (table.Rows.Count > 0)
          {
            table.RemoveRow(0);
          }
          foreach (var x in Lasts)
          {
            List<string> columns = new();
            columns.Add(DateTime.Now.ToString("HH\\:mm\\:ss"));          
            columns.Add(Math.Round(tableElapsedSecs, 0).ToString());
            columns.AddRange(x.Value);
            table.AddRow(columns.ToArray());

          }
          AnsiConsole.Write(table);
          lastTime = DateTime.Now;
        }

        //this.context.Refresh();

        //        int threadID = System.Threading.Thread.CurrentThread.ManagedThreadId;
        //        Console.WriteLine($"[{configID}] {numRowsAdded} {endRow} {posPerSecond} {time} {elapsedSecs} {numPositions} {totalLoss} {valueLoss} {valueAcc} {policyLoss} {policyAcc} {curLR}");
      } 
    }
  }
}