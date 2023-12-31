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
using System.IO;
using System.Collections.Generic;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base.OperatingSystem;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess;

using CeresTrain.TPG;
using CeresTrain.Trainer;
using CeresTrain.TPG.TPGGenerator;


#endregion

namespace CeresTrain.TrainData.TPGDatasets
{
  /// <summary>
  /// Static helper methods that facilitate creating TPG records from LC0 training data (v6).
  /// </summary>
  public static class TPGTorchDatasetComboHelpers
  {
    public static void TestOnTheFlyTPGGenerator(string dirTrainingTARs,
                                            string singleTrainingTARPath,
                                            Func<TPGRecord[], bool> bufferPostprocessor,
                                            TPGGeneratorOptions.DeblunderType deblunderType,
                                            bool allowFilterOutRepeatedPositions,
                                            bool rescoreTablebases,
                                            bool partiallyFilterObviousDrawsAndWins,
                                            int batchSize,
                                            bool verbose,
                                            int numConcurrentFiles,
                                            int positionSkipCount)
    {
#if KNOWLEDGE_DISTILLATION
      string NET1 = "801307";
#else
      string NET1 = null;
#endif

      if (singleTrainingTARPath != null && !File.Exists(singleTrainingTARPath))
      {
        throw new Exception("File not found: " + singleTrainingTARPath);
      }

      if (singleTrainingTARPath != null && new FileInfo(singleTrainingTARPath).Directory.Name.ToUpper() != new DirectoryInfo(dirTrainingTARs).Name.ToUpper())
      {
        throw new Exception("Mismatched directory: " + singleTrainingTARPath + " vs " + dirTrainingTARs);
      }

      int gpuID = SoftwareManager.IsLinux ? 0 : 1;
      NNEvaluator nNEvaluator = NET1 == null ? null : NNEvaluator.FromSpecification($"LC0:{NET1}", $"GPU:{gpuID}");

      TPGGeneratorOptions options = new TPGGeneratorOptions()
      {
        TargetCompression = CompressionLevel.Optimal,
        BatchSize = batchSize,

        // TODO: cleanup hardcoded path here
        CeresJSONFileName = SoftwareManager.IsLinux ? @"/raid/dev/Ceres/artifacts/release/net7.0/Ceres.json"
                                                : @"c:\dev\ceres\artifacts\release\net7.0\Ceres.json",
        Description = "Filled planes, deduplicated, deblundered, focus for 512b Apr 2021 (T60)",
        SourceDirectory = dirTrainingTARs,
        FilenameFilter = singleTrainingTARPath == null ? null : f => f.ToUpper() == singleTrainingTARPath.ToUpper(),

#if KNOWLEDGE_DISTILLATION
        AnnotationNNEvaluator = nNEvaluator, 
        AnnotationPostprocessor = delegate(in Position position,
                                           NNEvaluatorResult nnEvalResult,
                                           in EncodedTrainingPosition trainingPosition,
                                           ref TPGWriterNonPolicyTargetInfo nonPolicyTarget,
                                           ref CompressedPolicyVector? overridePolicyTarget)
        {
          // NOTE: returning false here is not yet supported
          if (nonPolicyTarget.Source == TPGWriterNonPolicyTargetInfo.TargetSourceInfo.Tablebase)
          {
            // Don't overwrite tablebase values, since they are definiitve.
            // TODO: should we try to remember the best tablebase move and reflect in policy below?
            // unclear, NN still has uncertainty. nonPolicyTarget.UNC = 0;
          }
          else
          {
            Interlocked.Add(ref numAccepted, 1);
#if NOT
            // This code replaces uncertainty based on Q from the teacher net.
            float searchQ = trainingPosition.PositionWithBoards.MiscInfo.InfoTraining.BestQ;
            float netQ = nnEvalResult.V;
            float uncertainty = Math.Min(1, Math.Abs(searchQ - netQ));
            nonPolicyTarget.UNC = uncertainty;
#else
            // Keep the uncertainty as measured by the original training network.
            // This method probably better because focuses on change induced by search,
            // rather than trying to learn difference between teacher net and net used in training.
            // WARNING: However if the teacher net and the training net were very different 
            //          then the above replacement would probably be better.
            float uncertainty = nonPolicyTarget.UNC;
#endif
            TPGWriterNonPolicyTargetInfo replacementNonPolicyTarget = new ()
            {
              Source = TPGWriterNonPolicyTargetInfo.TargetSourceInfo.Postprocessor,
              ResultWDL = (nnEvalResult.W, nnEvalResult.D, nnEvalResult.L),
              BestWDL = (nnEvalResult.W, nnEvalResult.D, nnEvalResult.L),
              IntermediateWDL = (nnEvalResult.W, nnEvalResult.D, nnEvalResult.L), // ??
              MLH = TPGRecordEncoding.ToMLHForNet(nnEvalResult.M),
              UNC = uncertainty,
              DeltaQForwardAbs = default // ??
            };
            nonPolicyTarget = replacementNonPolicyTarget;
          }
          CompressedPolicyVector policy = default;
          unsafe
          {
            policy = nnEvalResult.Policy;
            // to replicate training data:
            //   CompressedPolicyVector.Initialize(ref policy, trainingPosition.Policies.ProbabilitiesPtr, false, true);
          }
          overridePolicyTarget = policy;

          return true;
        },
#endif
        AcceptRejectAnnotater = delegate (EncodedTrainingPositionGame game, int positionIndex, in Position position)
        {
          if (!partiallyFilterObviousDrawsAndWins)
          {
            return true;
          }
          else
          {
            // Filter 50% of obvious wins and 30% of obvious draws,
            // where obvious is based on search result with agreeement
            // of the value head from the nerual network used in training.
            var infoTraining = game.PositionTrainingInfoAtIndex(positionIndex);
            bool isObviousWin = Math.Abs(infoTraining.BestQ) > 0.75f
                          && Math.Abs(infoTraining.BestQ - infoTraining.OriginalQ) < 0.10f;
            bool isObviousDraw = Math.Abs(infoTraining.BestQ) < 0.02f
                          && Math.Abs(infoTraining.BestQ - infoTraining.OriginalQ) < 0.03f
                          && infoTraining.BestD > 0.95f;

            if (isObviousWin || isObviousDraw)
            {
              int hashMod = position.GetHashCode() % 10;
              if (isObviousWin && hashMod < 6)
              {
                return false;
              }

              if (isObviousDraw && hashMod < 3)
              {
                return false;
              }
              return true;
            }
          }
#if NOT
          if (isObviousWin)
            Console.WriteLine("W   " + infoTraining.BestQ + " " + infoTraining.OriginalQ + "  " + gamePositions[positionIndex].PositionWithBoards.FENForHistoryBoard(0));
          if (isObviousDraw)
            Console.WriteLine("D   " + infoTraining.BestQ + " " + infoTraining.OriginalQ + "  " + gamePositions[positionIndex].PositionWithBoards.FENForHistoryBoard(0));
          else
            Console.WriteLine("?   " + infoTraining.BestQ + " " + infoTraining.OriginalQ + " " + gamePositions[positionIndex].PositionWithBoards.FENForHistoryBoard(0));
//          Console.WriteLine(isObvious);
#endif
          return true;

          return !position.PieceExists(Pieces.WhiteQueen) && !position.PieceExists(Pieces.BlackQueen)
    && !position.PieceExists(Pieces.WhiteRook) && !position.PieceExists(Pieces.BlackRook);
          //              && !position.PieceExists(Pieces.WhiteBishop) && !position.PieceExists(Pieces.BlackBishop);

          //          if (Math.Abs(gamePositions.Span[positionIndex].PositionWithBoards.MiscInfo.InfoTraining.BestQ) > 0.5f
          // && position.CalcZobristHash(PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98) % 2 == 0
          //  && numAccepted < 2_000_000)
          //            return false;
        },

        OutputFormat = TPGGeneratorOptions.OutputRecordFormat.TPGRecord, // note: possibly update FillInHistoryPlanes below
        FillInHistoryPlanes = false, // doesn't make sense with TPG

        RescoreWithTablebase = rescoreTablebases,
        EmitPlySinceLastMovePerSquare = TPGRecord.EMIT_PLY_SINCE_LAST_MOVE_PER_SQUARE,
        PositionSkipCount = positionSkipCount,
        NumThreads = numConcurrentFiles,
        NumConcurrentSets = numConcurrentFiles,

        //            RescoreWithTablebase = true,
        NumPositionsTotal = 40 * 1024L * 2048L * batchSize, // arbitrary large

        Deblunder = deblunderType,
        DeblunderThreshold = 0.06f,

        TargetFileNameBase = null,
        BufferPostprocessorDelegate = (records) =>
        {
          //Console.WriteLine(DateTime.Now + " got " + records.Length);
          bufferPostprocessor?.Invoke(records);
          return true;
        }
      };

      // Create the generator and run
      TrainingPositionGenerator tpg = new(options, verbose);
      tpg.RunGeneratorLoop();
    }

    public static IEnumerator<TPGRecord[]> GeneratorTPGRecordsViaGeneratorFromV6(string dirTrainingTARs,
                                                                                 string singleTrainingTARPath,
                                                                                 int batchSize,
                                                                                 TPGGeneratorOptions.DeblunderType deblunderType,
                                                                                 bool allowFilterOutRepeatedPositions,
                                                                                 bool rescoreTablebases,
                                                                                 bool partiallyFilterObviousDrawsAndWins,
                                                                                 bool verbose,
                                                                                 bool singleThreadSingleSetMember = false)
    {
      DateTime startTime = DateTime.Now;
      bool allDone = false;

      List<TPGRecord[]> pendingBatches = new();
      void BackgroundFill()
      {
        int batchCount = 0;

        byte[] bufferBackgroundThread = new byte[Marshal.SizeOf<TPGRecord>() * batchSize];

        // Alternate between two distinct buffers
        const int READAHEAD_COUNT = 20;
        const int POSITION_SKIP_COUNT = 20;
        const int NUM_CONCURRENT_STREAMS_PER_READER = 10;

        try
        {
          DoBackgroundFill(dirTrainingTARs, singleTrainingTARPath, batchSize, deblunderType, allowFilterOutRepeatedPositions,
                           rescoreTablebases, partiallyFilterObviousDrawsAndWins, singleThreadSingleSetMember, pendingBatches,
                           READAHEAD_COUNT, POSITION_SKIP_COUNT, NUM_CONCURRENT_STREAMS_PER_READER, verbose);
        }
        catch (Exception e)
        {
          Console.WriteLine("Exception encountered in background fill operation: " + e);
        }

      }

      Task.Run(BackgroundFill);
      Random r = new Random();
      while (!allDone)
      {
        TPGRecord[] ret = default;
        bool found = false;
        lock (pendingBatches)
        {
          if (pendingBatches.Count > 0)
          {
            // Pick and remove an entry at random
            int index = (int)r.NextInt64(pendingBatches.Count);
            ret = pendingBatches[index];
            pendingBatches.RemoveAt(index);

            found = true;
          }
        }

        if (found)
        {
          yield return ret;
        }
        else
        {
          Thread.Sleep(20);
        }
      }
    }

    private static void DoBackgroundFill(string dirTrainingTARs,
                                         string singleTrainingTARPath,
                                         int batchSize,
                                         TPGGeneratorOptions.DeblunderType deblunderType,
                                         bool allowFilterOutRepeatedPositions,
                                         bool rescoreTablebases,
                                         bool partiallyFilterObviousDrawsAndWins,
                                         bool singleThreadSingleSetMember,
                                         List<TPGRecord[]> pendingBatches,
                                         int READAHEAD_COUNT,
                                         int POSITION_SKIP_COUNT,
                                         int NUM_CONCURRENT_STREAMS,
                                         bool verbose)
    {

      TestOnTheFlyTPGGenerator(dirTrainingTARs, singleTrainingTARPath, delegate (TPGRecord[] records)
      {
        while (true)
        {
          int count = 0;
          lock (pendingBatches)
          {
            count = pendingBatches.Count;
          }

          if (count < READAHEAD_COUNT)
          {
            if (records.Length != batchSize)
            {
              throw new Exception("Incorrectly sized buffer returned " + records.Length + " expected " + batchSize);
            }

            // Make a copy and enqueue.
            TPGRecord[] copy = new TPGRecord[records.Length];
            Array.Copy(records, copy, records.Length);
            lock (pendingBatches)
            {
              pendingBatches.Add(copy);
            }
            break;
          }
          else
          {
            // Pause until the readahead buffer is not full anymore.
            Thread.Sleep(30);
          }
        }
        return true;
      }, deblunderType, allowFilterOutRepeatedPositions, rescoreTablebases, partiallyFilterObviousDrawsAndWins, batchSize, verbose,
        NUM_CONCURRENT_STREAMS, POSITION_SKIP_COUNT);
    }


    static long numAccepted = 0;


  }
}
