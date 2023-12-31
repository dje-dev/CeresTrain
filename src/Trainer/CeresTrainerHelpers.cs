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
  /// Various miscellaneous static helper methods for CeresTrainer.
  /// </summary>
  internal static partial class CeresTrainerHelpers
  {
    /// Extracts win/draw/loss logits from specified index in an array
    /// and converts to normalized probabilities
    /// in a numerically stable way and with checking for NaN.
    public static (float winProb, float drawProb, float lossProb) ExtractValueWDL(Half[] predictionsValueAll, int i)
      => WDLFromLogits((float)predictionsValueAll[i * 3 + 0],
                       (float)predictionsValueAll[i * 3 + 1],
                       (float)predictionsValueAll[i * 3 + 2]);

    /// Extracts win/draw/loss logits from specified index in an array
    /// and converts to normalized probabilities
    /// in a numerically stable way and with checking for NaN.
    /// </summary>
    public static (float winProb, float drawProb, float lossProb) ExtractValueWDL(float[] predictionsValueAll, int i)
      => WDLFromLogits(predictionsValueAll[i * 3 + 0], predictionsValueAll[i * 3 + 1], predictionsValueAll[i * 3 + 2]);


    /// <summary>
    /// Converts win/draw/loss logits into normalized probabilities
    /// in a numerically stable way, with checking for NaN.
    /// </summary>
    /// <param name="winLogit"></param>
    /// <param name="drawLogit"></param>
    /// <param name="lossLogit"></param>
    /// <returns></returns>
    public static (float winProb, float drawProb, float lossProb) WDLFromLogits(float winLogit, float drawLogit, float lossLogit)
    {
      float max = MathF.Max(MathF.Max(winLogit, drawLogit), lossLogit);

      float win = MathF.Exp((float)winLogit - max);
      float draw = MathF.Exp((float)drawLogit - max);
      float loss = MathF.Exp((float)lossLogit - max);

      float sum = win + draw + loss;
      if (float.IsNaN(sum))
      {
        throw new Exception($"WDL (value head output) contained NaN.");
      }

      win /= sum;
      draw /= sum;
      loss /= sum;

      return (win, draw, loss);
    }


  }
}