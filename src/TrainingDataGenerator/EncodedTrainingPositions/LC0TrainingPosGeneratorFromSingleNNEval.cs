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

using Ceres.Base.Math;
using Ceres.Base.Misc;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.APIExamples;
using Ceres.Chess.Data.Nets;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  public class LC0TrainingPosGeneratorFromSingleNNEval
  {
    // We might permanently mark any self-generated positions
    // by setting the value of Unused2 in the training record.
    // const int UNUSED2_VALUE_MARKER = 255;

    public readonly NNEvaluator Evaluator;
    //    public readonly NNEvaluator EvaluatorCeres;

    public LC0TrainingPosGeneratorFromSingleNNEval(NNEvaluator nnEvaluator)
    {
      Evaluator = nnEvaluator;
      //      EvaluatorCeres = CeresNetworkTesting.Evaluator1;
    }


    public EncodedTrainingPosition GenerateTrainingPosition(in EncodedTrainingPosition position, bool verbose)
    {
      return GenerateTrainingPositionFromNNEval(Evaluator, position.Version, position.InputFormat, position.PositionWithBoards.MiscInfo.InfoTraining.InvarianceInfo,
                                                position.ToPositionWithHistory(8), false, verbose);
    }

    public EncodedTrainingPosition GenerateNextTrainingPosition(in EncodedTrainingPosition startPos, MGMove moveToPlay, bool overrideResultToBeWin = false, bool verbose = false)
    {
      // Get current position with history and the played move.
      PositionWithHistory currentPos = startPos.ToPositionWithHistory(8);
      MGPosition thisPos = currentPos.FinalPosition.ToMGPosition;

      MGPosition nextPos = thisPos;
      nextPos.MakeMove(moveToPlay);

      PositionWithHistory nextPosition = new PositionWithHistory(currentPos);
      nextPosition.AppendPosition(nextPos, moveToPlay);

      return GenerateTrainingPositionFromNNEval(Evaluator, startPos.Version, startPos.InputFormat, startPos.PositionWithBoards.MiscInfo.InfoTraining.InvarianceInfo,
                                                nextPosition, overrideResultToBeWin, verbose);
    }


    public static EncodedTrainingPosition GenerateTrainingPositionFromNNEval(NNEvaluator evaluator, int version, int inputFormat, byte invarianceInfo, PositionWithHistory searchPosition, bool overrideResultToBeWin, bool verbose)
    {
      // Run neural net evaluation of this position (locking for concurrency control).
      NNEvaluatorResult evalResult;
      lock (evaluator)
      {
        evalResult = evaluator.Evaluate(searchPosition);
      }

      return EncodedTrainingPositionExtractor.ExtractFromNNEvalResult(evalResult, version, inputFormat, invarianceInfo, searchPosition, overrideResultToBeWin, verbose);
    }


    #region Helper methods


    /// <summary>
    /// Returns if the position looks like a terminal blunder, i.e. 
    /// a not obviously lost position where the chosen move was not optimal.
    /// </summary>
    /// <param name="lastPos"></param>
    /// <returns></returns>
    public static bool TrainingPosWasForcedMovePossiblySeriousBlunder(in EncodedPositionWithHistory lastPos, float thresholdBlunder = 0.7f)
    => TrainingPosWasForcedMovePossiblySeriousBlunder(in lastPos, out _, thresholdBlunder);


    /// <summary>
    /// Returns if the position looks like a terminal blunder, i.e. 
    /// a not obviously lost position where the chosen move was not optimal.
    /// </summary>
    /// <param name="lastPos"></param>
    /// <returns></returns>
    public static bool TrainingPosWasForcedMovePossiblySeriousBlunder(in EncodedPositionWithHistory lastPos,
                                                                      out PositionWithHistory finalPos,
                                                                      float thresholdBlunder = 0.7f)

    {
      ref readonly EncodedPositionEvalMiscInfoV6 lastInfo = ref lastPos.MiscInfo.InfoTraining;

      MGPosition mgPos = lastPos.FinalPosition.ToMGPosition;
      MGMove mgMove = lastPos.PlayedMove;
      mgPos.MakeMove(mgMove);

      finalPos = lastPos.ToPositionWithHistory(8);
      finalPos.AppendPosition(mgPos, mgMove);

      bool wasWinLoss = Math.Abs(lastInfo.ResultQ) == 1;
      bool wasForcedBlunder = lastInfo.NotBestMove;
      float THRESHOLD_BLUNDER = 0.7f;
      bool lastPosWasFarFromActualResult = Math.Abs(lastInfo.BestQ - lastInfo.ResultQ) > thresholdBlunder;
      return wasWinLoss && wasForcedBlunder && lastPosWasFarFromActualResult;
    }
    
    #endregion

    #region Statistics collecting 
    public static void CollectStats()
    {
      // Only 2 terminal blunders in the whole (small) file! These are my regenerated KP files.
      CollectStats(@"e:\t80_kp_zst.tar\training-run1-test80-20221227-1917.zst.tar");

      // Terminal blunder frequency: 0.051164918% of games: 5.642497%   (3304999 positions)
      CollectStats(@"d:\tar\t75\training-run3-test75-20210510-2326.tar");

      // Terminal blunder frequency: 0.07043422% of games: 7.7228265%   (7599999 positions)
      CollectStats(@"d:\tar\t80_new\training-run1-test80-20230103-1817.tar");

      // Terminal blunder frequency: 0.08012603% of games: 8.569773%   (7934999 positions)
      CollectStats(@"d:\tar\t81_pre_smolgen\training-run1-test80-20230706-0817.tar");

      // Terminal blunder frequency: 0.076225564% of games: 8.350562%   (4344999 positions)
      CollectStats(@"d:\tar\t81\training-run1-test80-20230902-1317.tar");

      //Terminal blunder frequency: 0.07761699% of games: 8.638951%   (5043999 positions)
      CollectStats(@"d:\tar\t81\training-run1-test80-20230814-1917.tar");
    }


    public static void CollectStats(string testTARFileName)
    {
      Console.WriteLine("\r\n" + testTARFileName);
      CountTerminalBlundersUnexplored(testTARFileName);
    }



    public static void CountTerminalBlundersUnexplored(string sourceTARFileName)
    {
      LC0TrainingPosGeneratorFromSingleNNEval generator = new LC0TrainingPosGeneratorFromSingleNNEval(NNEvaluator.FromSpecification(RegisteredNets.Aliased["T80"].NetSpecificationString, "GPU:0"));

      int countPosSeen = 0;
      int countTerminalBlunders = 0;
      int countGamesSeen = 0;

      foreach (Memory<EncodedTrainingPosition> game in new EncodedTrainingPositionReaderTAR(sourceTARFileName).EnumerateGames())
      {
        countGamesSeen++;
        countPosSeen += game.Length;
        if (game.Length > 1 && TrainingPosWasForcedMovePossiblySeriousBlunder(in game.Span[^1].PositionWithBoards))
        {
          countTerminalBlunders++;
        }

        const int SHOW_STATS_FREQUENCY = 1000;
        if (countPosSeen % SHOW_STATS_FREQUENCY == SHOW_STATS_FREQUENCY - 1)
        {
          Console.WriteLine($"Terminal blunder frequency of positions: {100.0f * countTerminalBlunders / countPosSeen}% "
                         + $", of games: {100.0f * countTerminalBlunders / countGamesSeen}%   ({countPosSeen} terminal blunder positions)");
        }
      }
      Console.WriteLine("total blunders: " + countTerminalBlunders);
    }


    public static void TestRegenerateAllPositions(string sourceTARFileName)
    {
      LC0TrainingPosGeneratorFromSingleNNEval generator = new LC0TrainingPosGeneratorFromSingleNNEval(NNEvaluator.FromSpecification(RegisteredNets.Aliased["T80"].NetSpecificationString, "GPU:0"));

      foreach (Memory<EncodedTrainingPosition> game in new EncodedTrainingPositionReaderTAR(sourceTARFileName).EnumerateGames())
      {
        const bool PROCESS_ALL = true;
        if (PROCESS_ALL || game.Length > 1 && TrainingPosWasForcedMovePossiblySeriousBlunder(in game.Span[^1].PositionWithBoards))
        {
          for (int i = 0; i < game.Length; i++)
          {
            int POS_INDEX_TO_TEST = i;
            bool FORCE_WON = i == game.Length - 1 && TrainingPosWasForcedMovePossiblySeriousBlunder(in game.Span[i].PositionWithBoards);

            if (FORCE_WON)
            {
              Console.WriteLine("FORCING WON");
            }

            const bool GEN_NEXT = false;

            EncodedTrainingPosition generatedNextPos = GEN_NEXT ?
                generator.GenerateNextTrainingPosition(in game.Span[POS_INDEX_TO_TEST], game.Span[POS_INDEX_TO_TEST].PositionWithBoards.PlayedMove, FORCE_WON, false)
              : generator.GenerateTrainingPosition(in game.Span[0], false);

            if (i == game.Length - 1)
            {
              // Special case, we don't have a correctNextPos in the training data. Just dump generated value compared against itself.
              ObjUtils.CompareAndPrintObjectFields(generatedNextPos.PositionWithBoards.MiscInfo.InfoTraining,
                                                   generatedNextPos.PositionWithBoards.MiscInfo.InfoTraining);
            }
            else
            {
              EncodedTrainingPosition tarPos = game.Span[GEN_NEXT ? POS_INDEX_TO_TEST + 1 : POS_INDEX_TO_TEST];

              float maxProbAbsDiff = MathUtils.MaxAbsoluteDifference(generatedNextPos.Policies.ProbabilitiesSpan.ToArray(), tarPos.Policies.ProbabilitiesSpan.ToArray());
              float vDiff = MathF.Abs(generatedNextPos.PositionWithBoards.MiscInfo.InfoTraining.BestQ - tarPos.PositionWithBoards.MiscInfo.InfoTraining.BestQ);


              if (maxProbAbsDiff < 0.40f && vDiff < 0.15f)
              {
                // Just dump one line if everything looked similar between the training data and regenerated data.
                Console.WriteLine(i + " Policy/value within 0.4, 0.15 same value, omit detail dump.");
              }
              else
              {
                // Dump extensive detail if the training data and regenerated data looked very different.
                Console.WriteLine($"Max absolute policy difference: {maxProbAbsDiff}");
                Console.WriteLine("ours: " + generatedNextPos.Policies.BestMove + " TAR: " + tarPos.Policies.BestMove);

                Console.WriteLine(game.Span[POS_INDEX_TO_TEST].PositionWithBoards.FinalPosition);
                generatedNextPos.Policies.DumpProb();
                Console.WriteLine();
                tarPos.Policies.DumpProb();

                ObjUtils.CompareAndPrintObjectFields(generatedNextPos.PositionWithBoards.MiscInfo.InfoPosition,
                                                     tarPos.PositionWithBoards.MiscInfo.InfoPosition);
                Console.WriteLine();
                ObjUtils.CompareAndPrintObjectFields(generatedNextPos.PositionWithBoards.MiscInfo.InfoTraining,
                                                     tarPos.PositionWithBoards.MiscInfo.InfoTraining);
              }
            }
          }

        }
      }
    }

    #endregion

  }
}
