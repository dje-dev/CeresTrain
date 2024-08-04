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
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Threading;

using Zstandard.Net;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using System.Threading.Tasks;

#endregion

namespace CeresTrain.TPG.TPGGenerator
{
  /// <summary>
  /// Manages writing of training positions to files,
  /// supporting concurrent writers and splitting across multiple files in a set
  /// (to enhance speed and shuffling of data).
  /// 
  /// Two output formats are supported, either LC0 v6 training positions
  /// or TPGRecords (Ceres format containing position in a format directly consumed by Ceres neural networks).
  /// </summary>
  internal class TrainingPositionWriter
  {
    /// <summary>
    /// The size of the batches which are assembled before
    /// being written written to disk.
    /// </summary>
    int BUFFER_SIZE = 4096;

    public readonly string OutputFileNameBase;
    public readonly TPGGeneratorOptions.OutputRecordFormat OutputFormat;
    public readonly NNEvaluator Evaluator;
    public readonly TrainingPositionGenerator.PositionPostprocessor EvaluatorPostprocessor;
    public readonly long TotalNumPositionsToBeWritten;
    public readonly bool EmitPlySinceLastMovePerSquare;
    public readonly bool EmitHistory;

    public int NumPositionsWritten => numBuffersWritten * BUFFER_SIZE;


    int[] numRecordsWritten;

    public long numPositionsRejectedByPostprocessor = 0;

    readonly object lockObj = new();
    int numBuffersWritten = 0;
    int[] countsInBuffer;

    /// <summary>
    /// Buffers (when output format is EncodedTrainingPosition)
    /// </summary>
    EncodedTrainingPosition[][] buffers;

    byte[][][] bufferPliesSinceLastPieceMoveBySquare;

    public TrainingPositionWriterNonPolicyTargetInfo[][] buffersTargets;
    CompressedPolicyVector?[][] buffersOverridePolicies;

    Stream[] outStreams;

    readonly object[] writingBuffersLocks;
    Func<TPGRecord[], bool> BatchPostprocessorDelegate;

    bool shutdown = false;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="outputFileNameBase">base file name for output, or null if not written to files</param>
    /// <param name="numSets"></param>
    /// <param name="outputFormat"></param>
    /// <param name="useZstandard"></param>
    /// <param name="compressionLevel"></param>
    /// <param name="totalNumPositionsToBeWritten"></param>
    /// <param name="evaluator"></param>
    /// <param name="evaluatorPostprocessor"></param>
    /// <param name="batchPostprocessorDelegate">called with each TPGRecord[] generate (make copy of array if needed subsequently)</param>
    /// <param name="batchSize"></param>
    /// <param name="emitPlySinceLastMovePerSquare"></param>
    /// <param name="emitHistory"></param>
    /// <exception cref="Exception"></exception>
    /// <exception cref="ArgumentException"></exception>
    public TrainingPositionWriter(string outputFileNameBase, int numSets,
                                  TPGGeneratorOptions.OutputRecordFormat outputFormat,
                                  bool useZstandard, CompressionLevel compressionLevel,                                  
                                  long totalNumPositionsToBeWritten,
                                  NNEvaluator evaluator,
                                  TrainingPositionGenerator.PositionPostprocessor evaluatorPostprocessor,
                                  Func<TPGRecord[], bool> batchPostprocessorDelegate,
                                  int batchSize,
                                  bool emitPlySinceLastMovePerSquare,
                                  bool emitHistory = true)
    {
      BUFFER_SIZE = batchSize;

      if (totalNumPositionsToBeWritten % BUFFER_SIZE != 0)
      {
        throw new Exception("TotalNumPositions must be a multiple of " + BUFFER_SIZE);
      }

      if (outputFormat != TPGGeneratorOptions.OutputRecordFormat.TPGRecord && evaluatorPostprocessor != null)
      {
        throw new ArgumentException("evaluatorPostprocessor only supported with TPGOptions.OutputRecordFormat.TPGRecord");
      }

      numSets = (int)Math.Min(numSets, totalNumPositionsToBeWritten / batchSize);

      OutputFileNameBase = outputFileNameBase;
      TotalNumPositionsToBeWritten = totalNumPositionsToBeWritten;
      OutputFormat = outputFormat;
      Evaluator = evaluator;
      EvaluatorPostprocessor = evaluatorPostprocessor;
      BatchPostprocessorDelegate = batchPostprocessorDelegate;
      EmitPlySinceLastMovePerSquare = emitPlySinceLastMovePerSquare;
      EmitHistory = emitHistory;

      outStreams = outputFileNameBase == null ? null : new Stream[numSets];
      buffers = new EncodedTrainingPosition[numSets][];
      buffersTargets = new TrainingPositionWriterNonPolicyTargetInfo[numSets][];
      buffersOverridePolicies = new CompressedPolicyVector?[numSets][];
      numRecordsWritten = new int[numSets];
      bufferPliesSinceLastPieceMoveBySquare = new byte[numSets][][];
      writingBuffersLocks = new object[numSets];
      countsInBuffer = new int[numSets];

      for (int i = 0; i < numSets; i++)
      {
        // Allocate buffers for this set.
        writingBuffersLocks[i] = new object();
        buffers[i] = new EncodedTrainingPosition[BUFFER_SIZE];
        buffersTargets[i] = new TrainingPositionWriterNonPolicyTargetInfo[BUFFER_SIZE];
        buffersOverridePolicies[i] = new CompressedPolicyVector?[BUFFER_SIZE];

        if (emitPlySinceLastMovePerSquare)
        {
          bufferPliesSinceLastPieceMoveBySquare[i] = new byte[BUFFER_SIZE][];
          for (int b = 0; b < BUFFER_SIZE; b++)
          {
            bufferPliesSinceLastPieceMoveBySquare[i][b] = new byte[64];
          }
        }

        if (outputFileNameBase != null)
        {
          // Create output stream.
          string fn = outputFileNameBase + "_set" + i + (useZstandard ? ".zst" : ".gz");
          Stream outStream = new FileStream(fn, FileMode.Create);

          // NOTE: Level 11 is 15% slower to compress and 10% smaller in size compared to level 10
          int[] zStdCompressionEquivalents = new int[] { 11, 5, 0, 16 };
          Stream compressedStream = useZstandard ? new ZstandardStream(outStream, zStdCompressionEquivalents[(int)compressionLevel])
                                                 : new GZipStream(outStream, compressionLevel);
          outStreams[i] = compressedStream;
        }
      }
    }


    public void Write(int targetSetIndex, float minLegalMoveProbability, 
                      params ((EncodedTrainingPosition record, TrainingPositionWriterNonPolicyTargetInfo targetInfo, 
                      int indexMoveInGame, short[] indexLastMoveBySquares), bool validate)[] items)
    {
      // Take the lock on the buffer associated with this target set
      // so that we can't have two concurrent writes to the same target set.
      lock (buffersTargets[targetSetIndex])
      {
        foreach (var item in items)
        {
          Write(item.Item1.record, item.Item1.targetInfo, item.Item1.indexMoveInGame, item.Item1.indexLastMoveBySquares, minLegalMoveProbability, targetSetIndex, item.validate);
        }
      }
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="record"></param>
    /// <param name="targetInfo"></param>
    /// <param name="indexMoveInGame"></param>
    /// <param name="indexLastMoveBySquares"></param>
    /// <param name="minLegalMoveProbability"></param>
    /// <param name="targetSetIndex"></param>
    /// <param name="emitMoves"></param>
    public void Write(in EncodedTrainingPosition record, in TrainingPositionWriterNonPolicyTargetInfo targetInfo,
                      int indexMoveInGame, short[] indexLastMoveBySquares, float minLegalMoveProbability,
                      int targetSetIndex, bool validate)
    {
      if (shutdown)
      {
        return;
      }

      if (validate)
      {
        // Just before writing validate record integrity one more time.
        EncodedTrainingPosition.ValidateIntegrity(record.InputFormat, record.Version,
                                                  in record.PositionWithBoards, in record.Policies,
                                                  "TrainingPositionGenerator postprocessing validity check failure: " + OutputFileNameBase);
      }

      lock (buffersTargets[targetSetIndex])
      {
        int thisBufferIndex = countsInBuffer[targetSetIndex];
        buffers[targetSetIndex][thisBufferIndex] = record;
        buffersTargets[targetSetIndex][thisBufferIndex] = targetInfo;

        if (EmitPlySinceLastMovePerSquare)
        {
          // Compute and buffer the number of moves since each square has seen a piece move.
          byte[] pliesSinceLastPieceMoveBySquare = bufferPliesSinceLastPieceMoveBySquare[targetSetIndex][thisBufferIndex];
          for (int s = 0; s < 64; s++)
          {
            pliesSinceLastPieceMoveBySquare[s] = TPGRecordEncoding.ToPliesSinceLastPieceMoveBySquare(indexMoveInGame, indexLastMoveBySquares[s]);
          }
        }

        countsInBuffer[targetSetIndex]++;

        if (countsInBuffer[targetSetIndex] == BUFFER_SIZE)
        {
          if (Evaluator == null)
          {
            // No NN evaluation needed, we can synchronously do the batch writing.
            ProcessWrite(buffers[targetSetIndex], buffersTargets[targetSetIndex], buffersOverridePolicies[targetSetIndex],
                         minLegalMoveProbability, bufferPliesSinceLastPieceMoveBySquare[targetSetIndex], targetSetIndex, validate);
          }
          else
          {
            // NN evaluation and postprocessing needed, most do asynchronously.

            // Extract the array with these positions so they are not overwritten
            // and create a new array to receive future positions which arrive
            // before the postprocessing is complete.
            EncodedTrainingPosition[] positions = buffers[targetSetIndex];
            buffers[targetSetIndex] = new EncodedTrainingPosition[BUFFER_SIZE];

            ProcessWrite(positions, buffersTargets[targetSetIndex], buffersOverridePolicies[targetSetIndex],
                         minLegalMoveProbability, bufferPliesSinceLastPieceMoveBySquare[targetSetIndex], targetSetIndex, validate);

#if NOT
Disabled for now. If the NN evaluator can't keep up, the set of pending Tasks grows without bound.
            // Spin off as a separate task.
            Task.Run(() => ProcessWrite(positions, targetSetIndex));
#endif
          }

          countsInBuffer[targetSetIndex] = 0;
        }
      }
    }


    /// <summary>
    /// Applies postprocessing to the specified array of positions.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="startIndexNonNNResult"></param>
    /// <param name="results"></param>
    /// <param name="nonPolicyTarget"></param>
    /// <param name="overridePolicyTarget"></param>
    /// <param name="indicesPositionsToOmit"></param>
    /// <exception cref="NotImplementedException"></exception>
    void Postprocessor(EncodedTrainingPosition[] positions,
                      int startIndexNonNNResult,
                      Memory<NNEvaluatorResult> results,
                      TrainingPositionWriterNonPolicyTargetInfo[] nonPolicyTarget,
                      CompressedPolicyVector?[] overridePolicyTarget,
                      HashSet<int> indicesPositionsToOmit)
    {
      MGMoveList movesList = new MGMoveList();
      int numToProcess = results.Length;
      Span<EncodedMove> movesNotPresent = stackalloc EncodedMove[CompressedPolicyVector.NUM_MOVE_SLOTS];
      Span<NNEvaluatorResult> resultsSpan = results.Span;

      for (int i = 0; i < numToProcess; i++)
      {
        NNEvaluatorResult nnResult = resultsSpan[i];
        Position thisPosition = positions[i + startIndexNonNNResult].PositionWithBoards.FinalPosition;

        // Always reject positions with missing BestQ/BestD.
        EncodedPositionEvalMiscInfoV6 trainingMiscInfo = positions[i + startIndexNonNNResult].PositionWithBoards.MiscInfo.InfoTraining;
        bool missingBestInfo = float.IsNaN(trainingMiscInfo.BestQ + trainingMiscInfo.BestD);

        bool acceptPos;
        if (missingBestInfo)
        {
          Console.WriteLine("Warning: NaN found for BestQ or BestD in a training position: " + thisPosition.FEN);
          acceptPos = false;
        }
        else
        {
          ref CompressedPolicyVector? thisOverridePolicyTarget = ref overridePolicyTarget[i + startIndexNonNNResult];

          // Call the supplied delegate with this position and associated NN evaluation (if Evaluator was specified).
          acceptPos = EvaluatorPostprocessor(thisPosition,
                                             nnResult,
                                             ref positions[i + startIndexNonNNResult],
                                             ref nonPolicyTarget[i + startIndexNonNNResult],
                                             ref thisOverridePolicyTarget);

          if (thisOverridePolicyTarget.HasValue)
          {
            // Build an array of moves which are legal but do not already appear in the CompressedPolicyVector.
            int numNotPresent = 0;

            movesList.Clear();
            MGMoveGen.GenerateMoves(thisPosition.ToMGPosition, movesList);
            for (int m = 0; m < movesList.NumMovesUsed; m++)
            {
              EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(movesList.MovesArray[m]);
              if (thisOverridePolicyTarget.Value.IndexOfMove(encodedMove) == -1)
              {
                movesNotPresent[numNotPresent++] = encodedMove;
              }
            }

            if (numNotPresent > 0)
            {
              throw new NotImplementedException("Need to write a method to append movesNotPresent to CompressedPolicyVector with zero probability");
            }
          }
        }

        if (!acceptPos)
        {
          indicesPositionsToOmit.Add(i + startIndexNonNNResult);
        }
      }

      Interlocked.Add(ref numPositionsRejectedByPostprocessor, indicesPositionsToOmit.Count);
    }


    /// <summary>
    /// Runs the postprocessor on specified array of positions.
    /// </summary>
    /// <param name="targetSetIndex"></param>
    /// <param name="positions"></param>
    /// <param name="batch"></param>
    /// <returns></returns>
    HashSet<int> PostprocessBatch(int targetSetIndex, EncodedTrainingPosition[] positions, EncodedPositionBatchFlat batch)
    {
      HashSet<int> indicesPositionsToOmit = new();
      if (Evaluator != null)
      {
        // Start out with assumption that policy target is not overridden.
        Array.Clear(buffersOverridePolicies[targetSetIndex], 0, BUFFER_SIZE);

        Evaluator.EvaluateOversizedBatch(batch, (int startIndex, Memory<NNEvaluatorResult> results) 
                                        => Postprocessor(positions, startIndex, results,
                                                         buffersTargets[targetSetIndex],
                                                         buffersOverridePolicies[targetSetIndex],
                                                         indicesPositionsToOmit));
      }
      else
      {
        Postprocessor(positions, 0, null, buffersTargets[targetSetIndex],
                      buffersOverridePolicies[targetSetIndex], indicesPositionsToOmit);
      }
      return indicesPositionsToOmit;
    }



    [ThreadStatic]
    static TPGRecord[] tpgRecordsBuffer = null;

    /// <summary>
    /// Returns a new array of TPGRecords converted from the specified input data.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="includeHistory"></param>
    /// <param name="targetInfos"></param>
    /// <param name="targetPolicyOverrides"></param>
    /// <param name="minLegalMoveProbability"></param>
    /// <param name="pliesSinceLastPieceMoveBySquare"></param>
    /// <param name="emitMoves"></param>
    /// <returns></returns>
    unsafe TPGRecord[] ConvertedTPGRecords(EncodedTrainingPosition[] positions,
                                           bool includeHistory,
                                           TrainingPositionWriterNonPolicyTargetInfo[] targetInfos,
                                           CompressedPolicyVector?[] targetPolicyOverrides,
                                           float minLegalMoveProbability,
                                           byte[][] pliesSinceLastPieceMoveBySquare,
                                           bool validate)
    {
      if (tpgRecordsBuffer == null)
      {
        tpgRecordsBuffer = new TPGRecord[positions.Length];
      }

      TPGRecord[] bufferForParallelThreads = tpgRecordsBuffer;

      // Convert in parallel. 
      const int MAX_PARALLELISM = 8; // not too large, since there is already a lot of concurrency in these classes.
      Parallel.For(0, positions.Length, new ParallelOptions() { MaxDegreeOfParallelism = MAX_PARALLELISM }, i =>
      {
        // Convert into TPG record format.
        TPGRecordConverter.ConvertToTPGRecord(in positions[i], includeHistory, in targetInfos[i], targetPolicyOverrides?[i],
                                              minLegalMoveProbability, ref bufferForParallelThreads[i],
                                              pliesSinceLastPieceMoveBySquare?[i], EmitPlySinceLastMovePerSquare,
                                              targetInfos[i].ForwardSumNegativeBlunders, targetInfos[i].ForwardSumPositiveBlunders,
                                              targetInfos[i].PriorPositionWinP,
                                              targetInfos[i].PriorPositionDrawP,
                                              targetInfos[i].PriorPositionLossP, validate);
      });

      return tpgRecordsBuffer;
    }



    /// <summary>
    /// Write a specified buffer of positions to the specified output set.
    /// Handles:
    ///   - calling postprocessor (if any)
    ///   - converting to TPG records (if TGP is the target format type)
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="positionsTargets"></param>
    /// <param name="targetPolicyOverrides"></param>
    /// <param name="minLegalMoveProbability"></param>
    /// <param name="pliesSinceLastPieceMoveBySquare"></param>
    /// <param name="targetSetIndex"></param>
    /// <exception cref="NotImplementedException"></exception>
    private unsafe void ProcessWrite(EncodedTrainingPosition[] positions,
                                     TrainingPositionWriterNonPolicyTargetInfo[] positionsTargets,
                                     CompressedPolicyVector?[] targetPolicyOverrides,
                                     float minLegalMoveProbability,
                                     byte[][] pliesSinceLastPieceMoveBySquare, int targetSetIndex, 
                                     bool validate)
    {
      // Take a lock so that we insure we can't have two concurrent
      // postprocesses/writes to the same target set.
      lock (writingBuffersLocks[targetSetIndex])
      {
        if (EvaluatorPostprocessor != null)
        {
          // Create a big batch out of all the positions collected.
          EncodedPositionBatchFlat batchFlat = null;

          if (Evaluator != null)
          {
            batchFlat = new EncodedPositionBatchFlat(positions, BUFFER_SIZE, EncodedPositionType.PositionOnly, true);

            // Set all the moves.
            // TODO: possibly this can be omitted, since there seems to be logic to
            //       fill this in if needed downstream in preparation for evaluation.
            MGMoveList[] moves = new MGMoveList[batchFlat.NumPos];
            for (int i = 0; i < positions.Length; i++)
            {
              Position thisPosition = positions[i].PositionWithBoards.FinalPosition;

              MGMoveList thisPosMoves = new MGMoveList();
              MGMoveGen.GenerateMoves(thisPosition.ToMGPosition, thisPosMoves);
              moves[i] = thisPosMoves;
            }
            batchFlat.Moves = moves;
          }

          // Launch a postprocessor to update.
          HashSet<int> indicesPositionsToOmit = PostprocessBatch(targetSetIndex, positions, batchFlat);


          if (indicesPositionsToOmit.Count > 0)
          {
            // Create new array of only the selected positions
            int numToKeep = positions.Length - indicesPositionsToOmit.Count;
            EncodedTrainingPosition[] selectedPositions = new EncodedTrainingPosition[numToKeep];
            int countAdded = 0;
            for (int i = 0; i < positions.Length; i++)
            {
              if (!indicesPositionsToOmit.Contains(i))
              {
                selectedPositions[countAdded++] = positions[i];
              }
            }

            positions = selectedPositions;
          }
        }

        if (OutputFormat == TPGGeneratorOptions.OutputRecordFormat.TPGRecord)
        {
          // Convert to TPG.
          TPGRecord[] convertedToTPG = ConvertedTPGRecords(positions, EmitHistory, positionsTargets, targetPolicyOverrides, minLegalMoveProbability,
                                                           pliesSinceLastPieceMoveBySquare, validate);

          if (BatchPostprocessorDelegate != null)
          {
            bool processedOK = BatchPostprocessorDelegate(convertedToTPG);
            if (!processedOK)
            {
              throw new NotImplementedException("Shutdown not yet implemented");
            }
          }

          if (outStreams != null)
          {
            // Write bytes to the file (via pinned memory).
            GCHandle tpgDataHandle = GCHandle.Alloc(convertedToTPG, GCHandleType.Pinned);
            ReadOnlySpan<byte> bufferAsBytes = MemoryMarshal.Cast<TPGRecord, byte>(convertedToTPG);
            outStreams[targetSetIndex].Write(bufferAsBytes);
            tpgDataHandle.Free();            
          }
        }
        else if (OutputFormat == TPGGeneratorOptions.OutputRecordFormat.EncodedTrainingPos)
        {
          if (!EmitHistory)
          {
            throw new NotImplementedException();
          }

          if (outStreams != null)
          {
            // Write bytes to the file (via pinned memory).
            GCHandle posDataHandle = GCHandle.Alloc(positions, GCHandleType.Pinned);
            ReadOnlySpan<byte> bufferAsBytes = MemoryMarshal.Cast<EncodedTrainingPosition, byte>(positions);
            outStreams[targetSetIndex].Write(bufferAsBytes);
            posDataHandle.Free();
          }
        }
        else
        {
          throw new NotImplementedException("Unsupported OutputFormat of " + OutputFormat);
        }

        // Record fact that records actually written.
        numRecordsWritten[targetSetIndex] += positions.Length;

        // Increment done count.
        Interlocked.Increment(ref numBuffersWritten);
      }
    }


    /// <summary>
    /// Return string description.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<TrainingPositionWriter {OutputFormat} to {OutputFileNameBase} with {TotalNumPositionsToBeWritten} per set";
    }


    public void Shutdown()
    {
      shutdown = true;
      if (outStreams != null)
      {
        lock (lockObj)
        {
          if (outStreams != null)
          {
            for (int i = 0; i < outStreams.Length; i++)
            {
              outStreams[i].Dispose();
              outStreams[i] = null;
            }
          }
          outStreams = null;
        }
      }
    }


  }


}