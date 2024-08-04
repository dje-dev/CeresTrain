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
using System.Linq;
using System.Threading;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Collections.Concurrent;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NNEvaluators.LC0DLL;

using CeresTrain.TPG;
using CeresTrain.TPG.TPGGenerator;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  /// <summary>
  /// An ITPGBatchGenerator which produces batches of TPG records
  /// where the Positions are derived a specified generator, and the
  /// training targets (value, policy, etc.) are taken from tablebases.
  /// 
  /// Tablebase server for lichess.org, based on shakmaty-syzygy
  /// (including CURL interface and including mainline calculations).
  ///   https://github.com/lichess-org/lila-tablebase 
  ///   https://github.com/lichess-org/lila-tablebase/blob/develop/src/main.rs#L465 [mainline]
  ///   curl http://tablebase.lichess.ovh/standard/mainline?fen=8/5P2/K7/1p6/6P1/p7/8/2k5_w_-_-_0_0
  /// Online interactive lookup position in TB:
  ///   https://syzygy-tables.info/?fen=8/3P4/4P2p/8/5K2/7p/3k4/8_w_-_-_0_1
  /// </summary>
  public class TablebaseTPGBatchGenerator : ITPGBatchGenerator
  {
    /// <summary>
    /// Description of type of positions generated.
    /// </summary>
    public readonly string Description;

    /// <summary>
    /// Number of positions to be returned in each batch.
    /// </summary>
    public readonly int BatchSize;

    /// <summary>
    /// Number of worker threads to launch in background to prepare positions.
    /// </summary>
    public readonly int NumWorkerThreads;

    /// <summary>
    /// Provided function which is called to generate positions.
    /// </summary>
    public readonly Func<Position> PosGenerator;

    /// <summary>
    /// If some DTZ information is found missing (e.g. incomplete 7 man tablebases),
    /// if we should process the position anyway (using fill-in/guess for policy info that may be missing).
    /// </summary>
    public readonly bool SucceedIfIncompleteDTZInformation;


    TrainingPositionWriter writer;
    int maxQueueLength;
    long numRead = 0;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="description"></param>
    /// <param name="posGenerator"></param>
    /// <param name="batchSize"></param>
    /// <param name="numWorkerThreads"></param>
    public TablebaseTPGBatchGenerator(string description, Func<Position> posGenerator, 
                                      bool succeedIfIncompleteDTZInformation,
                                      int batchSize = 2048, int numWorkerThreads = 5)
    {
      if (numWorkerThreads <= 0)
      {
        throw new ArgumentException("must be greater than zero", nameof(numWorkerThreads));
      }

      if (batchSize <= 0)
      {
        throw new ArgumentException("must be greater than zero", nameof(batchSize));
      }

      Description = description;
      NumWorkerThreads = numWorkerThreads;
      PosGenerator = posGenerator;
      BatchSize = batchSize;
      SucceedIfIncompleteDTZInformation = succeedIfIncompleteDTZInformation;
    }


    public TPGRecord[] GetBatch()
    {
      if (!haveStarted)
      {
        const int DEFAULT_QUEUE_LENGTH = 4;
        Start(DEFAULT_QUEUE_LENGTH);
      }

      TPGRecord[] batch;
      while (!PendingRecords.TryDequeue(out batch))
      {
        Thread.Sleep(30);
      }
      return batch;
    }

    public IEnumerable<TPGRecord[]> Enumerator()
    {
      while (true)
      {
        yield return GetBatch();
      }
    }


    #region Interface method implementations

    bool haveStarted;

    /// <summary>
    /// Begins generation of batches in the background and filling of queue of pending batches.
    /// </summary>
    /// <param name="maxQueueLength"></param>
    public void Start(int maxQueueLength)
    {
      if (!haveStarted)
      {
        this.maxQueueLength = maxQueueLength;
        StartGeneratorThreads();
        haveStarted = true;
      }
    }


    /// <summary>
    /// Accessor to the underlying queue of pending batches generated.
    /// </summary>
    public ConcurrentQueue<TPGRecord[]> PendingBatchQueue => PendingRecords;

    #endregion

    public readonly int MaxQueueLength;

    /// <summary>
    /// Queue of pending batches generated.
    /// </summary>
    public ConcurrentQueue<TPGRecord[]> PendingRecords = new();


    bool PostProcessor(TPGRecord[] records)
    {
      while (PendingRecords.Count > maxQueueLength)
      {
        Thread.Sleep(100);
      }

#if DEBUG
      Array.ForEach(records, tpgRecord => TPGRecordValidation.Validate(tpgRecord));
#endif

      Interlocked.Add(ref numRead, records.Length);

      // Make a safe private copy of records.
      TPGRecord[] copy = new TPGRecord[records.Length];
      Array.Copy(records, copy, copy.Length);

      // Enqueue records.
      PendingRecords.Enqueue(copy);

      return true;
    }



    public void StartGeneratorThreads()
    {
      const int NUM_WRITE = int.MaxValue;
      const bool WRITE_TO_FILE = false;
      const bool EMIT_HISTORY = false;
      int TARGET_NUM_TPG() => BatchSize * NumWorkerThreads;

      int random = Environment.TickCount % 100_000;
      string outFileName = WRITE_TO_FILE ? "KP_" + random + ".tpg" : null;

      writer = new TrainingPositionWriter(outFileName, NumWorkerThreads, TPGGeneratorOptions.OutputRecordFormat.TPGRecord,
                                          true, System.IO.Compression.CompressionLevel.Optimal, TARGET_NUM_TPG(),
                                          null, null, PostProcessor, BatchSize, false, EMIT_HISTORY);

      ISyzygyEvaluatorEngine tbEvaluator = ISyzygyEvaluatorEngine.DefaultEngine;
      List<Task> tasksList = new List<Task>();
      for (int i = 0; i < NumWorkerThreads; i++)
      {
        int thisIndex = i;
        tasksList.Add(Task.Run(() =>
        {
          try
          {
            RunPositionGeneration(writer, tbEvaluator, thisIndex, NUM_WRITE, SucceedIfIncompleteDTZInformation);
          }
          catch (Exception exc)
          {
            ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Exception in TablebaseTPGBatchGenerator: " + exc.ToString());
          }
        }));
      }

      tasks = tasksList.ToArray();
    }

    Task[] tasks;
    volatile bool shouldShutdown = false;


    /// <summary>
    /// Shuts down all background worker thread and release resources.
    /// </summary>
    public void Shutdown()
    {
      if (tasks != null)
      {
        shouldShutdown = true;

        // Drain queue.
        bool threadsAllDone = false;
        while (!threadsAllDone)
        {
          PendingBatchQueue.Clear();
          threadsAllDone = Task.WaitAll(tasks.ToArray(), 100);
        }

        writer.Shutdown();
        tasks = null;
      }
    }


    private void RunPositionGeneration(TrainingPositionWriter writer,
                                       ISyzygyEvaluatorEngine tbEvaluator,
                                       int indexInSet,
                                       int numToWrite,
                                       bool succeedIfIncompleteDTZData)
    {
      int numWritten = 0;
      while (!shouldShutdown && numWritten < numToWrite)
      {
        Position pos = PosGenerator();

        // Skip terminal positions.
        if (pos.CalcTerminalStatus() != GameResult.Unknown)
        {
          continue;
        }

        if (pos.SideToMove == SideType.Black)
        {
          pos = pos.Reversed;
        }

        bool generated = TrainingRecordFromTablebase.GenerateTrainingRecordFromTablebase(in pos, tbEvaluator, succeedIfIncompleteDTZData,
                                                                                         out EncodedTrainingPosition etp,
                                                                                         out TrainingPositionWriterNonPolicyTargetInfo targetInfo);

        if (generated)
        {
          writer.Write(in etp, targetInfo, 0, null, CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE, indexInSet, true);
          numWritten++;
        }

      }
    }

  }
}
