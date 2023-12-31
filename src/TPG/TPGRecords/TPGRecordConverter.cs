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
using System.Runtime.InteropServices;
using System.Threading.Tasks;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using CeresTrain.TPG.TPGGenerator;

#endregion

namespace CeresTrain.TPG
{
  /// <summary>
  /// Static helper class to convert EncodedTrainingPosition to corresponding TPGRecord.
  /// </summary>
  public static class TPGRecordConverter
  {
    static bool IsInvalid(float f) => float.IsNaN(f) || float.IsInfinity(f);
    static bool IsInvalid((float w, float d, float l) item) => float.IsNaN(item.w + item.d + item.l)
                                                            || float.IsInfinity(item.w + item.d + item.l)
                                                            || item.w + item.d + item.l < 0.999
                                                            || item.w + item.d + item.l > 1.001;

#if TO_BE_RESTORED_MAYBE

// Probably best to first convert to EncodedTrainingPositionWithHistory and then call the existing method below

    public static unsafe void ConvertToTPGCombo(in Position position, int? targetSquareFromPriorMoveFromOurPerspective,
                                                bool isRepetition, in EncodedPolicyVector policies,
                                                ref TPGRecordCombo tpgRecordCombo)
    {
      // Write the moves into a temporary buffer
      Span<TPGMoveRecord> moveRecords = stackalloc TPGMoveRecord[TPGRecordCombo.MAX_SQUARES_AND_MOVES];
      MGMoveList moves = TPGMoveRecord.WriteMoves(in position, moveRecords,
                                                  in policies, tpgRecordCombo.Policies);

      // Write the squares (with pieces) into temporary buffer
      Span<TPGSquareRecord> squareRecords = stackalloc TPGSquareRecord[TPGRecord.MAX_SQUARES];
      Span<byte> squareIndices = stackalloc byte[64];

      TPGSquareRecord.WritePosPieces(in position, isRepetition, targetSquareFromPriorMoveFromOurPerspective, squareRecords, squareIndices);

      // Normally there will be enough room to write all legal moves,
      // all piece squares must be written for sure (up to 32) and there are only 96 total slots.
      int MAX_MOVES_TO_WRITE = TPGRecordCombo.MAX_SQUARES_AND_MOVES - position.PieceCount;

      const bool FORCE_64_32 = false;
      if (FORCE_64_32)
      {
        MAX_MOVES_TO_WRITE = 64;
      }
      int NUM_MOVES_TO_WRITE = Math.Min(MAX_MOVES_TO_WRITE, moves.NumMovesUsed);


      for (int i = 0; i < NUM_MOVES_TO_WRITE; i++)
      {
        // Copy over the move record
        tpgRecordCombo.SquaresAndMoves[i].MoveRecord = moveRecords[i];

#if NOT
        // NOTE: This is disabled. It makes training very significantly slower.
        //       Instead better to leave empty so Transformer can distinguish types of records.
        // Also copy over associated piece record
        int fromSquareIndex = moveRecords[i].GetFromSquare().SquareIndexStartA1;
        tpgRecordCombo.SquaresAndMoves[i].SquareRecord = squareRecords[squareIndices[fromSquareIndex]];
#endif
        //Console.WriteLine(i + " move: " + moveRecords[i] + " square: " + squareRecords[squareIndices[fromSquareIndex]]);

      }
      //Console.WriteLine();
      // Subsequently write the squares (with pieces)
      int START_SQUARES = FORCE_64_32 ? 64 : NUM_MOVES_TO_WRITE;
      for (int i = START_SQUARES; i < START_SQUARES + position.PieceCount; i++)
      {
        tpgRecordCombo.SquaresAndMoves[i].SquareRecord = squareRecords[i - START_SQUARES];
        //Console.WriteLine(i + " " + tpgRecordCombo.SquaresAndMoves[i].SquareRecord);
      }

    }
#endif

    /// <summary>
    /// Converter from EncodedTrainingPosition to TPGRecord (fast version for inference, not supporting setting of training target info).
    /// </summary>
    /// <param name="encodedPosToConvert"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public static TPGRecord ConvertedToTPGRecord(in EncodedPositionWithHistory encodedPosToConvert,
                                                 bool includeHistory,
                                                 Span<byte> pliesSinceLastPieceMoveBySquare = default)
    {
#if MOVES
        throw new NotImplementedException();
#endif

      // N.B. Some logic here is the same as in method below (ConvertToTPGRecord) and should be kept in sync.
      TPGRecord tpgRecord = default;

      // Write squares.
      ConvertToTPGRecordSquares(in encodedPosToConvert, includeHistory, default, null,
                                ref tpgRecord, pliesSinceLastPieceMoveBySquare, pliesSinceLastPieceMoveBySquare != default);

      return tpgRecord;
    }


    public static unsafe void ConvertToTPGRecord(in EncodedPositionWithHistory encodedPosToConvert,
                                                 bool includeHistory,
                                                 Memory<MGMoveList> moves,
                                                 TrainingPositionWriterNonPolicyTargetInfo? targetInfo,
                                                 CompressedPolicyVector? policyVector,
                                                 float minLegalMoveProbability,
                                                 ref TPGRecord tpgRecord,
                                                 Span<byte> pliesSinceLastPieceMoveBySquare,
                                                 bool emitPlySinceLastMovePerSquare)
    {
      // N.B. Some logic here is the same as in method above (ConvertedToTPGRecord) and should be kept in sync.

      // Clear out any prior values.
      tpgRecord = default;

      // Write squares.
#if DISABLED_MIRROR
      // TODO: we mirror the position here to match expectation of trained net based on 
      //       LC0 training data (which is mirrored). Someday undo this mirroring in training and then can remove here.
      EncodedPositionWithHistory encodedPosToConvertMirrored = encodedPosToConvert.Mirrored;
#endif
      ConvertToTPGRecordSquares(in encodedPosToConvert, includeHistory, moves, targetInfo, ref tpgRecord, pliesSinceLastPieceMoveBySquare, emitPlySinceLastMovePerSquare);

      // Convert the values unrelated to moves and squares
      if (targetInfo != null)
      {
        ConvertToTPGEvalInfo(targetInfo.Value, ref tpgRecord);
      }

      if (policyVector is not null)
      {
        ConvertToTPGRecordPolicies(in policyVector, minLegalMoveProbability, ref tpgRecord);
      }

#if MOVES
      if (emitMoves)
      {
        Position finalPos = encodedPosToConvert.FinalPosition;
        if (!finalPos.IsWhite)
        {
          finalPos = finalPos.Reversed;
        }

        TPGMoveRecord.WriteMoves(finalPos.Mirrored, tpgRecord.MovesSetter, default, default);
      }
#endif

#if DEBUG
      const bool validate = true;
#else
      const bool validate = false;
#endif
      // Randomly validate some even in non-debug mode.
      //const int VALIDATE_PCT = 1;
      //bool validate = (trainingPos.PositionWithBoards.GetPlanesForHistoryBoard(0).GetHashCode() % 100) < VALIDATE_PCT;
      if (validate)
      {
        // Run validation.
        TPGRecordValidation.Validate(in encodedPosToConvert, in tpgRecord, policyVector is not null);
      }

    }


    static bool haveInitialized = false;

    const int MAX_TPG_RECORDS_PER_BUFFER = 4096;

    [ThreadStatic]
    static TPGRecord[] tempTPGRecords;

    /// <summary>
    /// Converts input positions defined as IEncodedPositionFlat 
    /// into raw square and move bytes used by the TPGRecord.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="lastMovePliesEnabled"></param>
    /// <param name="emitMoves"></param>
    /// <param name="mgPos"></param>
    /// <param name="squareBytesAll"></param>
    /// <param name="moveBytesAll"></param>
    public static void ConvertPositionsToRawSquareBytes(IEncodedPositionBatchFlat positions,
                                                         bool includeHistory,
                                                        Memory<MGMoveList> moves,
                                                        bool lastMovePliesEnabled,
                                                        out MGPosition[] mgPos,
                                                        out byte[] squareBytesAll,
                                                        out short[] legalMoveIndices)
    {
      short[] legalMoveIndicesInternal = new short[positions.NumPos * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST];
      legalMoveIndices = legalMoveIndicesInternal;

      // Get all positions from input batch.
      // TODO: Improve efficiency, these array materializations are expensive.
      Memory<EncodedPositionWithHistory> positionsFlat = positions.PositionsBuffer;
      if (positions.PositionsBuffer.IsEmpty)
      {
        throw new Exception("PositionsBuffer not initialized, EncodedPositionBatchFlat.RETAIN_POSITIONS_INTERNALS needs to be set true");
      }
      mgPos = positions.Positions.ToArray();
      byte[] pliesSinceLastMoveAllPositions = positions.LastMovePlies.ToArray();

      int offsetRawBoardBytes = 0;
      squareBytesAll = new byte[positions.NumPos * Marshal.SizeOf<TPGSquareRecord>() * 64];
      byte[] squareBytesAllCopy = squareBytesAll;

      // Determine each position and copy converted raw board bytes into rawBoardBytesAll.
      // TODO: for efficiency, avoid doing this if the NN evaluator does not need raw bytes
      Parallel.For(0, positions.NumPos, i =>

      //for (int i = 0; i < positions.NumPos; i++)
      {
        if (tempTPGRecords == null)
        {
          tempTPGRecords = new TPGRecord[MAX_TPG_RECORDS_PER_BUFFER];
        }

        if (!lastMovePliesEnabled)
        {
          // Disable any values possibly passed for last used plies since they are not to be used.
          pliesSinceLastMoveAllPositions = null;
        }

        TPGRecord tpgRecord = default;
        Span<byte> thesePliesSinceLastMove = pliesSinceLastMoveAllPositions == null ? default : new Span<byte>(pliesSinceLastMoveAllPositions, i * 64, 64);

        ConvertToTPGRecord(in positionsFlat.Span[i], includeHistory, moves, null, null, float.NaN,
                           ref tpgRecord, thesePliesSinceLastMove, lastMovePliesEnabled);
        tempTPGRecords[i] = tpgRecord;

        const bool VALIDITY_CHECK = true;
        if (VALIDITY_CHECK && pliesSinceLastMoveAllPositions != null)
        {
          tpgRecord = CheckPliesSinceLastMovedCorrect(tpgRecord);
        }

        // Extract as bytes.
        tpgRecord.CopySquares(squareBytesAllCopy, i * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD);

#if MOVES
        tpgRecord.CopyMoves(moveBytesAll, i * TPGRecord.MAX_MOVES * Marshal.SizeOf<TPGMoveRecord>());
#endif

        TPGRecordMovesExtractor.ExtractLegalMoveIndicesForIndex(tempTPGRecords, moves, legalMoveIndicesInternal, i);
      });
    }



    private static TPGRecord CheckPliesSinceLastMovedCorrect(TPGRecord tpgRecord)
    {
      throw new NotImplementedException();
    }


    public static unsafe void ConvertToTPGRecord(in EncodedTrainingPosition trainingPos,
                                                 bool includeHistory,
                                                 in TrainingPositionWriterNonPolicyTargetInfo targetInfo,
                                                 CompressedPolicyVector? overridePolicyVector,
                                                 float minLegalMoveProbability,
                                                 ref TPGRecord tpgRecord,
                                                 Span<byte> pliesSinceLastPieceMoveBySquare,
                                                 bool emitPlySinceLastMovePerSquare,
                                                 bool emitMoves)
    {
      trainingPos.ValidateIntegrity("Validate in ConvertToTPGRecord");

      // Clear out any prior values.
      tpgRecord = default;

      // Convert policies.
      if (overridePolicyVector is not null)
      {
        throw new NotImplementedException(); // see the else below
        ConvertToTPGRecordPolicies(in overridePolicyVector, minLegalMoveProbability, ref tpgRecord);
      }
      else
      {
        // Note that ConvertToTPGRecordPolicies is called first.
        // This will initialize the tpgRecord.Policy which is then referenced in WriteMoves below.
        ConvertToTPGRecordPolicies(in trainingPos, minLegalMoveProbability, ref tpgRecord);

#if MOVES
        if (emitMoves)
        {
          TPGMoveRecord.WriteMoves(in thisPosition, tpgRecord.MovesSetter, tpgRecord.Policy, tpgRecord.PolicyForMoves);
        }
#endif

      }

      // Convert the values unrelated to moves and squares
      ConvertToTPGEvalInfo(in targetInfo, ref tpgRecord);

      // Write squares.
      ConvertToTPGRecordSquares(trainingPos.PositionWithBoards, includeHistory, default, targetInfo, ref tpgRecord,
                                pliesSinceLastPieceMoveBySquare, emitPlySinceLastMovePerSquare);

#if DEBUG
      const bool validate = TPGRecord.NUM_SQUARES == 64;; // TODO: Generalize this for <64
#else
      const bool validate = false;
      // Randomly validate some even in non-debug mode.
      //const int VALIDATE_PCT = 1;
      //bool validate = (trainingPos.PositionWithBoards.GetPlanesForHistoryBoard(0).GetHashCode() % 100) < VALIDATE_PCT;
#endif
      if (validate)
      {
        // Run validation.
        TPGRecordValidation.Validate(in trainingPos.PositionWithBoards, ref tpgRecord, overridePolicyVector is not null);
      }
    }


    internal static unsafe void ConvertToTPGRecordPolicies(in CompressedPolicyVector? policyVector,
                                                           float minLegalMoveProbability,
                                                           ref TPGRecord tpgRecord)
    {
      if (policyVector is null)
      {
        throw new ArgumentNullException(nameof(policyVector));
      }

      int count = 0;
      foreach ((EncodedMove move, float probability) in policyVector.Value.ProbabilitySummary())
      {
        if (count < TPGRecord.MAX_MOVES)
        {
          float probAdjusted = MathF.Max(minLegalMoveProbability, probability);
          tpgRecord.PolicyValues[count] = (Half)probAdjusted;
          tpgRecord.PolicyIndices[count] = (short)move.IndexNeuralNet;
          count++;
        }
      }

      PadPolicies(ref tpgRecord, count);
    }


    /// <summary>
    /// Postprocesses policy array in a TPGRecord to set unused slots
    /// such that future scatter operations across the full array will be correct
    /// (replicate last entry).
    /// </summary>
    /// <param name="tpgRecord"></param>
    /// <param name="count"></param>
    unsafe static void PadPolicies(ref TPGRecord tpgRecord, int count)
    {
      if (count < TPGRecord.MAX_MOVES)
      {
        // Replicate the last entry into every remaining position
        // so that scattering across all slots will result in correct policy.
        short lastIndex = tpgRecord.PolicyIndices[count - 1];
        Half lastValue = tpgRecord.PolicyValues[count - 1];
        while (count < TPGRecord.MAX_MOVES)
        {
          tpgRecord.PolicyValues[count] = lastValue;
          tpgRecord.PolicyIndices[count] = lastIndex;
          count++;
        }
      }
    }


    internal static unsafe void ConvertToTPGRecordPolicies(in EncodedTrainingPosition trainingPos, float minLegalMoveProbability, ref TPGRecord tpgRecord)
    {
      // TODO: speed up. Check two at a time within loop for better parallelism?
      float* probabilitiesSource = &trainingPos.Policies.ProbabilitiesPtr[0];

      float incrementFromValuesSetAtMin = 0;
      int count = 0;
      for (int i = 0; i < 1858; i++)
      {
        if (count < TPGRecord.MAX_MOVES)
        {
          float prob = probabilitiesSource[i];
          if (prob >= 0)
          {
            if (prob < minLegalMoveProbability)
            {
              incrementFromValuesSetAtMin += minLegalMoveProbability - prob;
              prob = minLegalMoveProbability;
            }

            tpgRecord.PolicyIndices[count] = (short)i;
            tpgRecord.PolicyValues[count] = (Half)prob;
            count++;
          }
        }
      }

      // Restore "sum to 1.0" property if necessary.
      const float MAX_INCREMENT_ALLOWED_BEFORE_RENORMALIZE = 0.005f; // for efficiency ignore if very small deviation from 1.0
      if (incrementFromValuesSetAtMin > MAX_INCREMENT_ALLOWED_BEFORE_RENORMALIZE)
      {
        float multiplier = 1.0f / (1 + incrementFromValuesSetAtMin);
        for (int i = 0; i < count; i++)
        {
          float current = (float)tpgRecord.PolicyValues[i];
          tpgRecord.PolicyValues[i] = (Half)(current * multiplier);
        }
      }

      PadPolicies(ref tpgRecord, count);
    }



    public static unsafe void ConvertToTPGRecordSquares(in EncodedPositionWithHistory posWithHistory,
                                                        bool includeHistory,
                                                        Memory<MGMoveList> moves,
                                                        in TrainingPositionWriterNonPolicyTargetInfo? targetInfo,
                                                        ref TPGRecord tpgRecord,
                                                        Span<byte> pliesSinceLastPieceMoveBySquare,
                                                        bool emitPlySinceLastMovePerSquare)
    {
      static Position GetHistoryPosition(in EncodedPositionWithHistory historyPos, int index, in Position? fillInIfEmpty)
      {
        Position pos = historyPos.HistoryPositionIsEmpty(index) ? default
                                                                : historyPos.HistoryPosition(index);

#if NOT
        // NOTE: This is only to keep compatability with previously written TPG files.
        //       Someday consider backing this out (also in decode methods).
        pos = pos.Mirrored; 
#endif

        if (pos.PieceCount == 0 && fillInIfEmpty != null)
        {
          pos = fillInIfEmpty.Value;
        }

        // Put position in same perspective as final position (index 0).
        if (pos.IsWhite != (index % 2 == 0))
        {
          pos = pos.Reversed;
        }
        return pos;
      }

      tpgRecord.IsWhiteToMove = posWithHistory.MiscInfo.WhiteToMove ? (byte)1 : (byte)0;

      const bool FILL_IN = true;
      Position thisPosition = GetHistoryPosition(in posWithHistory, 0, null);
      Position historyPos1 = GetHistoryPosition(in posWithHistory, 1, FILL_IN ? thisPosition : null);
      Position historyPos2 = GetHistoryPosition(in posWithHistory, 2, FILL_IN ? historyPos1 : null);
      Position historyPos3 = GetHistoryPosition(in posWithHistory, 3, FILL_IN ? historyPos2 : null);
      Position historyPos4 = GetHistoryPosition(in posWithHistory, 4, FILL_IN ? historyPos3 : null);
      Position historyPos5 = GetHistoryPosition(in posWithHistory, 5, FILL_IN ? historyPos4 : null);
      Position historyPos6 = GetHistoryPosition(in posWithHistory, 6, FILL_IN ? historyPos5 : null);
      Position historyPos7 = GetHistoryPosition(in posWithHistory, 7, FILL_IN ? historyPos6 : null);

      if (!includeHistory)
      {
        // TODO: make more efficient
        if (FILL_IN)
        {
          historyPos1 = historyPos2 = historyPos3 = historyPos4 = historyPos5 = historyPos6 = historyPos6 = historyPos7 = thisPosition;
        }
        else
        {
          historyPos1 = historyPos2 = historyPos3 = historyPos4 = historyPos5 = historyPos6 = historyPos6 = historyPos7 = default;
        }
      }

      // Write squares.
      TPGSquareRecord.WritePosPieces(in thisPosition, in historyPos1, in historyPos2, in historyPos3,
                                     in historyPos4, in historyPos5, in historyPos6, in historyPos7,
                                     tpgRecord.Squares, pliesSinceLastPieceMoveBySquare, emitPlySinceLastMovePerSquare);


#if DEBUG
      const bool validate = true;
#else
      // Possibly randomly validate some even in non-debug mode.
      const int VALIDATE_PCT = 0;
      bool validate = VALIDATE_PCT > 0 && TPGRecord.NUM_SQUARES == 64 && thisPosition.CalcZobristHash(PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98, false) % 100 < VALIDATE_PCT;
#endif
      if (validate)
      {
        TPGRecordValidation.ValidateHistoryReachability(in tpgRecord);
        TPGRecordValidation.ValidateSquares(in posWithHistory, ref tpgRecord);
      }
    }



    /// <summary>
    /// Extracts TPGWriterNonPolicyTargetInfo into a TPGRecord.
    /// </summary>
    /// <param name="targetInfo"></param>
    /// <param name="tpgRecord"></param>
    /// <exception cref="Exception"></exception>
    internal static unsafe void ConvertToTPGEvalInfo(in TrainingPositionWriterNonPolicyTargetInfo targetInfo,
                                                     ref TPGRecord tpgRecord)
    {
      if (IsInvalid(targetInfo.ResultWDL)) throw new Exception("Bad ResultWDL " + targetInfo.ResultWDL);
      if (IsInvalid(targetInfo.BestWDL)) throw new Exception("Bad BestWDL " + targetInfo.BestWDL);
      if (IsInvalid(targetInfo.MLH)) throw new Exception("Bad MLH " + targetInfo.MLH);
      if (IsInvalid(targetInfo.DeltaQVersusV)) throw new Exception("Bad UNC " + targetInfo.DeltaQVersusV);

      tpgRecord.WDLResult[0] = targetInfo.ResultWDL.w;
      tpgRecord.WDLResult[1] = targetInfo.ResultWDL.d;
      tpgRecord.WDLResult[2] = targetInfo.ResultWDL.l;

      tpgRecord.WDLQ[0] = targetInfo.BestWDL.w;
      tpgRecord.WDLQ[1] = targetInfo.BestWDL.d;
      tpgRecord.WDLQ[2] = targetInfo.BestWDL.l;

      tpgRecord.MLH = targetInfo.MLH;
      tpgRecord.DeltaQVersusV = targetInfo.DeltaQVersusV;

#if NOT
// Old fill in directly from training position. Seemingly not needed now.
      else
      {
        ref readonly EncodedPositionEvalMiscInfoV6 evalInfo = ref trainingPos.PositionWithBoards.MiscInfo.InfoTraining;
        if (IsInvalid(evalInfo.PliesLeft))
        {
          Console.WriteLine("found PliesLefts NaN");
          tpgRecord.MLH = 100; // default fill-in
        }
        else
        {
          tpgRecord.MLH = evalInfo.PliesLeft;
        }

        // Set actual game result.
        if (IsInvalid(evalInfo.ResultWDL.w + evalInfo.ResultWDL.d + evalInfo.ResultWDL.l))
        {
          Console.WriteLine("found ResultWDL NaN");
          tpgRecord.WDLResult[0] = (float)1 / 3;
          tpgRecord.WDLResult[1] = (float)1 / 3;
          tpgRecord.WDLResult[2] = (float)1 / 3;
        }
        else
        {
          tpgRecord.WDLResult[0] = evalInfo.ResultWDL.w;
          tpgRecord.WDLResult[1] = evalInfo.ResultWDL.d;
          tpgRecord.WDLResult[2] = evalInfo.ResultWDL.l;
        }

        // Set engine Q at end of search (if available).
        if (IsInvalid(evalInfo.BestWDL.w + evalInfo.BestWDL.d + evalInfo.BestWDL.l))
        {
          Console.WriteLine("found BestWDL NaN, substitute ResultWDL");
          tpgRecord.WDLQ[0] = evalInfo.ResultWDL.w;
          tpgRecord.WDLQ[1] = evalInfo.ResultWDL.d;
          tpgRecord.WDLQ[2] = evalInfo.ResultWDL.l;
        }
        {
          tpgRecord.WDLQ[0] = evalInfo.BestWDL.w;
          tpgRecord.WDLQ[1] = evalInfo.BestWDL.d;
          tpgRecord.WDLQ[2] = evalInfo.BestWDL.l;
        }
      }
#endif
    }


  }

}
