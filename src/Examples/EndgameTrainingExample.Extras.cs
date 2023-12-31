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

namespace CeresTrain.Examples
{
  /// <summary>
  /// Additional miscellaneous methods for class EndgameTrainingExample.
  /// </summary>
  public partial class EndgameTrainingExample
  {
#if NOT
    /// <summary>
    /// Dumps only those positions for which specified net has incorrect value head classification.
    /// Each line consists of net value head result, correct TB result, FEN.
    /// </summary>
    /// <param name="netFN"></param>
    public static void DumpIncorrectValueHeadPositions(string netFN)
    {
      NNEvaluatorTorchsharp evaluator = CeresNetEvaluators.GetNNEvaluator(netFN, true);
      ISyzygyEvaluatorEngine tbEvaluator = ISyzygyEvaluatorEngine.DefaultEngine;

      while (true)
      {
        Position pos = randPosGenerator.GeneratePosition();

        NNEvaluatorResult evalResult = evaluator.Evaluate(in pos);
        int gameResultTablebase = tbEvaluator.ProbeWDLAsV(in pos);

        if (gameResultTablebase != evalResult.MostProbableGameResult)
        {
          Console.WriteLine(evalResult.MostProbableGameResult + " " + gameResultTablebase + " " + pos.FEN);
        }
      }
    }
#endif

  }

}
