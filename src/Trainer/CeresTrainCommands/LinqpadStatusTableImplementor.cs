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
using LINQPad;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Implementation of training status table for LINQPad.
  /// Uses DumpContainer to post status lines, as well as
  /// the LINQPAD ProgressBar.
  /// </summary>
  public class LINQPadStatusTableImplementor : TrainingStatusTableImplementor
  {
    Util.ProgressBar progressBar;

    DumpContainer dc;
    string title;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="maxPositions"></param>
    public LINQPadStatusTableImplementor(long maxPositions) : base(maxPositions)
    {
    }


    public override void SetTitle(string title)
    {
      Console.WriteLine("");
      Console.WriteLine(title);
      this.title = title;
    }

    public override void RunTraining(Action trainingLoop)
    {
      progressBar = new Util.ProgressBar(title).Dump();
      progressBar.HideWhenCompleted = true;
      progressBar.Percent = 0;

      trainingLoop();

      progressBar.Percent = 100;
    }

    float lastTotalLoss = 0;
    DateTime timeAtOnePercentDone = default;

    public override void UpdateInfo(int numLinesWritten, bool endRow, float posPerSecond,
                                    DateTime time, float elapsedSecs, long numPositions, float totalLoss, float valueLoss,
                                    float valueAcc, float policyLoss, float policyAcc, float curLR)
    {
      float pctDone = MathF.Min(100, 100 * numPositions / MaxPositions);
      if (timeAtOnePercentDone == default && pctDone >= 1)
      {
        timeAtOnePercentDone = DateTime.Now;
      }

      // Possibly show estimated time remaining to caption.
      if (pctDone >= 1.5f)
      {
        float secsPerPercent = (float)(DateTime.Now - timeAtOnePercentDone).TotalSeconds / (pctDone - 1);
        float secsRemaining = secsPerPercent * (100 - pctDone);
        string timeRemainingStr = $" (estimated {secsRemaining:F0} secs remaining...)";
        progressBar.Caption = timeRemainingStr;
      }

      progressBar.Percent = (int)MathF.Round(pctDone, 0);

      float deltaLoss = lastTotalLoss == 0 ? 0 : totalLoss - lastTotalLoss;

      string showStr = $"{time,-11:HH\\:mm\\:ss} {elapsedSecs,5:F1} {numPositions,12:N0}  "
       + $"{posPerSecond,7:N0}   {deltaLoss,7:F3} {totalLoss,7:F3} {policyLoss,7:F3} {valueLoss,7:F3} "
       + $"{100 * policyAcc,7:F2}% {100 * valueAcc,7:F2}%   {Math.Round(curLR, 6),-10:F6}";


      if (dc == null)
      {
        dc = new DumpContainer(showStr).Dump(title);
      }

      if (endRow)
      {
        dc.UpdateContent(showStr, true);
        dc = new DumpContainer(showStr).Dump(title);
        lastTotalLoss = totalLoss;

      }
      else
      {
        dc.UpdateContent(showStr);
      }
    }
  }
}
