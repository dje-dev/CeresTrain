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
using System.Diagnostics;
using System.Collections.Generic;
using System.Numerics.Tensors;

using Ceres.Chess.Positions;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen.Converters;

using CeresTrain.TPG.TPGGenerator;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  /// <summary>
  /// Converts a Position (which can be scored by tablebases) into a training record,
  /// filling in value and policy with reasonable training targets.
  /// </summary>
  public static class TrainingRecordFromTablebase
  {
    const int LC0_DATA_VERSION = 6;
    const int LC0_INPUT_FORMAT = 1;
    const int LC0_INVARIANCE_INFO = 32;


    /// <summary>
    /// Generates a single EncodedTrainingPosition from a position which can be scored by tablebases.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="tbEvaluator"></param>
    /// <param name="etp"></param>
    /// <param name="targetInfo"></param>
    /// <exception cref="Exception"></exception>
    public static bool GenerateTrainingRecordFromTablebase(in Position pos,
                                                           ISyzygyEvaluatorEngine tbEvaluator,
                                                           bool succeedIfIncompleteDTZ,
                                                           out EncodedTrainingPosition etp,
                                                           out TrainingPositionWriterNonPolicyTargetInfo targetInfo)
    {
      tbEvaluator.ProbeWDL(in pos, out SyzygyWDLScore score, out SyzygyProbeState tbResult);
      if (tbResult == SyzygyProbeState.Fail)
      {
        throw new Exception("Internal error: position not found in tablebase " + pos.FEN);
      }


      MGMove bestMove = tbEvaluator.CheckTablebaseBestNextMoveViaDTZ(in pos, out WDLResult result, out List<(MGMove, short)> moves, 
                                                                     out short dtz, false, succeedIfIncompleteDTZ);
      if (moves.Count == 1)
      {
        // If only one move, best move is that one move!
        bestMove = moves[0].Item1;
      }

      Debug.Assert(bestMove != default || succeedIfIncompleteDTZ);

      CompressedPolicyVector policyTarget = BuildPolicyTarget(score, moves);

      // SLOW, attempts to find actual DTM.
      // But this not possible with Syzygy, and is sometimes very wrong (confirmed with online tools that this is not possible).
      //List<(MGMove, short)> bestMovesWin = TestTrainingDataGeneratorTB.FindTablebaseWinMoves(tbEvaluator, pos.ToMGPosition);

      float q = score == SyzygyWDLScore.WDLLoss ? -1 : score == SyzygyWDLScore.WDLWin ? 1 : 0;
      etp = ExtractFromTablebase(LC0_DATA_VERSION, LC0_INPUT_FORMAT, LC0_INVARIANCE_INFO, new PositionWithHistory(pos), policyTarget, q);

      targetInfo = default;
      targetInfo.BestWDL = (q == 1 ? 1 : 0, q == 0 ? 1 : 0, q == -1 ? 1 : 0);
      targetInfo.Source = TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo.Tablebase;
      targetInfo.ResultDeblunderedWDL = targetInfo.BestWDL;
      targetInfo.ResultNonDeblunderedWDL = targetInfo.BestWDL;
      return true;
    }


    public static EncodedTrainingPosition ExtractFromTablebase(int version, int inputFormat, byte invarianceInfo,
                                                               PositionWithHistory searchPosition,
                                                               CompressedPolicyVector targetPolicy,
                                                               float q)
    {
      EncodedPositionWithHistory newPosHistory = default;

      if (true)
      {
        const bool FILL_HISTORY = true;
        Debug.Assert(searchPosition.Positions[0].SideToMove == SideType.White);
        newPosHistory.SetFromPosition(in searchPosition.Positions[0], FILL_HISTORY, SideType.White);
      }
      else
      {
        // NOTE: Inferior, does not set en passant (or do history fill at all).
        newPosHistory.SetFromSequentialPositions(searchPosition.Positions, false);
      }

      // Create the EncodedPolicyVector (start with -1's, then fill in with the actual populated values).
      EncodedPolicyVector epv = default;
      epv.InitilializeAllNegativeOne();
      unsafe
      {
        float* encodedProbs = epv.ProbabilitiesPtr;
        foreach ((EncodedMove Move, float Probability) entry in targetPolicy.ProbabilitySummary(0))
        {
          encodedProbs[entry.Move.IndexNeuralNet] = entry.Probability;
        }
      }

      int bestMoveIndex = epv.BestMove.IndexNeuralNet;

      float w = q > 0 ? 1 : 0;
      float d = q == 0 ? 1 : 0;
      float l = q < 0 ? 1 : 0;
      float m = 0; // Fill in someday?

      EncodedPositionEvalMiscInfoV6 trainingMiscInfo = new
        (
        invarianceInfo: invarianceInfo, depResult: default,
        rootQ: q, bestQ: q, rootD: d, bestD: d,
        rootM: m, bestM: m, pliesLeft: m,
        resultQ: q, resultD: d,
        playedQ: q, playedD: d, playedM: m,
        originalQ: q, originalD: d, originalM: m,
        numVisits: 1,
        playedIndex: (short)bestMoveIndex, bestIndex: (short)bestMoveIndex,
        unused1: default, unused2: default);

      EncodedTrainingPositionMiscInfo miscInfoAll = new(newPosHistory.MiscInfo.InfoPosition, trainingMiscInfo);
      newPosHistory.SetMiscInfo(miscInfoAll);

      EncodedTrainingPosition ret = new EncodedTrainingPosition(version, inputFormat, newPosHistory, epv);

      return ret;
    }


    public static CompressedPolicyVector BuildPolicyTarget(SyzygyWDLScore score, List<(MGMove, short)> moves)
    {
      Span<int> indices = stackalloc int[moves.Count];
      Span<float> probs = stackalloc float[moves.Count];

      int moveCount = 0;
      foreach ((MGMove, short) moveInfo in moves)
      {
        indices[moveCount] = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(moveInfo.Item1).IndexNeuralNet;
        probs[moveCount] = MoveInfoToPolicyScore(score, moveInfo.Item1, moveInfo.Item2);

        moveCount++;
      }

      Normalize(probs);

#if NOT_NEEDED_DONE_IN_TPG_WRITER
      for (int i = 0; i < scores.Length; i++)
      {
        if (scores[i] < CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE)
        {
          scores[i] = CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE;
        }
      }
      Normalize(scores, 1.0f);
#endif

      CompressedPolicyVector cpv = default;
      CompressedPolicyVector.Initialize(ref cpv, indices, probs, false); // TODO: this last argument can/probably should be removed! Nonsensical.
      return cpv;
    }


    static float MoveInfoToPolicyScore(SyzygyWDLScore score, MGMove move, int dtz)
    {
      if (dtz == ISyzygyEvaluatorEngine.DTZ_IF_DTZ_INDETERMINATE_WDL_KNOWN)
      {
        // The score is known and will be used below as primary influence on policy.
        // The actual DTZ is unknown but fortunately of secondary importance. Use a fill-in value.
        dtz = score == SyzygyWDLScore.WDLLoss ? -5 : 5;            
      }
      else if (dtz == ISyzygyEvaluatorEngine.DTZ_IF_DTZ_INDETERMINATE_WDL_UNKNOWN)
      {
        return FillInValueForUnknownMoveWDL(score, move);
      }

      if (score == SyzygyWDLScore.WDLLoss)
      {
        Debug.Assert(dtz <= 0);
        return 100 + -dtz * 2; // All lost, score similarly, but prefer avoiding zeroing slightly
      }
      else if (score == SyzygyWDLScore.WDLDraw
            || score == SyzygyWDLScore.WDLCursedWin
            || score == SyzygyWDLScore.WDLBlessedLoss)
      {
        return dtz == 0 ? 2000 : Math.Abs(dtz); // strong preference for draws (DTZ 0), otherwise slight preference for long DTZ
      }
      else if (score == SyzygyWDLScore.WDLWin)
      {
        if (dtz == 0)
        {
          return 40; // best possible if can immediately reset 50 move rule
        }
        else if (dtz < 0)
        {
          return Math.Max(0, -Math.Min(50, dtz) * 0.25f); // prefer pushing off reset of 50 move rule
        }
        else
        {
          float value = 1000 - Math.Min(50, dtz) * 10f;
          if (move.PromoteQueen)
          {
            value += 10_000;
          }
          else if (move.Capture)
          {
            value += 100;
          }
          return value;
        }
      }
      else
      {
        throw new NotImplementedException();
      }
    }

    /// <summary>
    /// Heuristic/guestimate score to be assigned to a move for which we don't know the WDL.
    /// </summary>
    /// <param name="currentPosWDL"></param>
    /// <param name="move"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    private static float FillInValueForUnknownMoveWDL(SyzygyWDLScore currentPosWDL, MGMove move)
    {
      if (currentPosWDL == SyzygyWDLScore.WDLLoss)
      {
        return 1; // policy doesn't much matter, we are lost no matter what
      }
      else // win or draw
      {
        Debug.Assert(move.IsPromotion); // Missing data only expected if promotion takes game into a new piece set.

        // Impossible to know correct values, just return values that prefer stronger promotions.
        // (But do not return a value not as high as would be used for possibly other moves which are known winning).
        if (move.PromoteQueen)
        {
          return 9000;
        }
        else if (move.PromoteRook)
        {
          return 900;
        }
        else if (move.PromoteBishop)
        {
          return 100;
        }
        else if (move.PromoteKnight)
        {
          return 100;
        }
        else
        {
          return 1; // not sure why this would happen, but just in case
        }
      }
    }


    /// <summary>
    /// Normalizes (make sum be 1.0) a set of values.
    /// </summary>
    /// <param name="values"></param>
    static void Normalize(Span<float> values)
    {
      float allSum = TensorPrimitives.Sum(values);

      // TODO: someday use Divide next
      //       but currently seems buggy!
      // TensorPrimitives.Divide(values, allSum, values);

      for (int i = 0; i < values.Length; i++)
      {
        values[i] /= allSum;
      }
    }

  }

}

