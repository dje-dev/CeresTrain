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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.UserSettings;


#endregion

namespace CeresTrain.TPG.TPGGenerator
{
  /// <summary>
  /// Generates TPG files containing postprocessed training positions
  /// (which have been deblundered, rescored, shuffled, converted to TPG, etc.)
  /// 
  /// NOTE: For deblunder, see https://github.com/LeelaChessZero/lc0/issues/1308.
  /// </summary>
  public class TrainingPositionGenerator
  {
    /// <summary>
    /// Optionally a postprocesssor delegate can be specified
    /// which allows arbitrary postprocessing (modification) 
    /// of the position record before it is written to the TPG.
    /// 
    /// Returns if the position should be accepted, otherwise it is passed over.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="nnEvalResult"></param>
    /// <param name="trainingPosition"></param>
    public delegate bool PositionPostprocessor(in Position position,
                                               NNEvaluatorResult nnEvalResult,
                                           ref EncodedTrainingPosition trainingPosition,
                                           ref TrainingPositionWriterNonPolicyTargetInfo nonPolicyTarget,
                                           ref CompressedPolicyVector? overridePolicyTarget);


    /// <summary>
    /// Specified options which control the parameters of the positino generation.
    /// </summary>
    public readonly TPGGeneratorOptions Options;

    TrainingPositionWriter writer;

    ISyzygyEvaluatorEngine eval;


    public long NumSkippedDueToModulus;
    public long NumSkippedDueToPositionFilter;
    public long NumSkippedDueToBadBestInfo;

    long numPosScanned = 0;
    long numPosSentToWriter = 0;
    long numDuplicatesSkipped = 0;

    public float PosGeneratedPerSec => (float)(writer.NumPositionsWritten / (DateTime.Now - Options.StartTime).TotalSeconds);

    readonly object consoleOutputLock = new();


    /// <summary>
    /// Constructor for a genetator which uses a specified set of options.
    /// </summary>
    /// <param name="options"></param>
    /// <param name="verbose"></param>
    public TrainingPositionGenerator(TPGGeneratorOptions options, bool verbose = true)
    {
      options.NumConcurrentSets = Math.Min(options.NumConcurrentSets, (int)( options.NumPositionsTotal / options.BatchSize));

      Options = options;
      Options.Validate();

      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Magenta, "Ceres Training Position Generator for v6 Data");

      // Get full set of files to random sample over. 
      List<string> filesToProcess = new List<string>(Directory.GetFiles(Options.SourceDirectory, "*.tar"));
      //.OrderByAscending(d => new FileInfo(d).GetLastWriteTime)
      if (Options.FilenameFilter != null)
      {
        filesToProcess = filesToProcess.Where(f => Options.FilenameFilter(f)).ToList();
      }

      if (filesToProcess.Count == 0)
      {
        throw new Exception($"No tar files in {Options.SourceDirectory}");
      }

      long sourceFilesSize = 0;
      foreach (string fn in filesToProcess)
      {
        sourceFilesSize += new FileInfo(fn).Length;
      }

      Options.SourceFilesSizeMB = sourceFilesSize / (float)(1024 * 1024);

      // Sort files to used deterministically (by their name).
      // Note however that due to threading, this does not guarantee clients
      // will see data from set of files in exactly the same order every time.
      filesToProcess.Sort((s1, s2) => s1.GetHashCode().CompareTo(s2.GetHashCode()));
      Options.FilesToProcess = new ConcurrentQueue<string>(filesToProcess);

      if (verbose)
      {
        // Dump options to Console.
        Options.Dump(Console.Out, false);
        Console.WriteLine();

        if (options.TargetFileNameBase != null)
        {
          using (TextWriter writer = new StreamWriter(options.TargetFileNameBase + ".tpg.options.txt"))
          {
            Options.Dump(writer, true);
          }
        }
      }
    }


    void Init()
    {
      string targetFNBase = Options.TargetFileNameBase == null ? null : Options.TargetFileNameBase + ".tpg";
      writer = new TrainingPositionWriter(targetFNBase, Options.NumConcurrentSets,
                                          Options.OutputFormat,
                                          Options.UseZstandard, Options.TargetCompression,                                         
                                          Options.NumPositionsTotal,
                                          Options.AnnotationNNEvaluator,
                                          Options.AnnotationPostprocessor,
                                          Options.BufferPostprocessorDelegate,
                                          Options.BatchSize,
                                          Options.EmitPlySinceLastMovePerSquare);

      if (Options.CeresJSONFileName == null)
      {
        CeresUserSettingsManager.LoadFromDefaultFile();
      }
      else
      {
        CeresUserSettingsManager.LoadFromFile(Options.CeresJSONFileName);
      }


      if (Options.RescoreWithTablebase)
      {
        eval = SyzygyEvaluatorPool.GetSessionForPaths(CeresUserSettingsManager.Settings.TablebaseDirectory);
        string tbDir = CeresUserSettingsManager.Settings.TablebaseDirectory;
        if (tbDir == null)
        {
          throw new NotImplementedException("Tablebase directory not specified in Ceres.json.");
        }
        else
        {
          eval = SyzygyEvaluatorPool.GetSessionForPaths(tbDir);
        }
      }
    }


    public void RunGeneratorLoop()
    {
      if (numPosScanned > 0)
      {
        throw new Exception("Generate loop already has run.");
      }

      Init();

      Console.WriteLine();
      Console.WriteLine($"LC0 game chunks will be sourced from {Options.FilesToProcess.Count} TAR files in directory {Options.SourceDirectory}");
      Console.WriteLine();

      // Seed the random number generator with time so each run generates new data.
      Random rand = new Random(DateTime.Now.Millisecond);

      // Launch all threads.
      int numThreadsToUse = Math.Min(Options.FilesToProcess.Count, Options.NumThreads);
      Task[] threads = new Task[numThreadsToUse];
      for (int i = 0; i < numThreadsToUse; i++)
      {
        Task task = new Task(() => RunGeneratorThread(Options.FilesToProcess, rand));
        task.Start();
        threads[i] = task;
      }

      Task.WaitAll(threads);

      // Write the summary file.
      if (Options.TargetFileNameBase != null)
      {
        using (TextWriter writerSummary = new StreamWriter(Options.TargetFileNameBase + ".tpg.summary.txt"))
        {
          writerSummary.WriteLine($"{PosGeneratedPerSec,6:F0}/sec,  "
                                + $"Scan: {numPosScanned,10:N0}  use: {numPosSentToWriter,9:N0}  skip_dups: {numDuplicatesSkipped,9:N0}  "
                                + $"Reject: {writer.numPositionsRejectedByPostprocessor,9:N0} "
                                + $"TBLook: {numTBLookup,9:N0}  TBFound: {numTBFound,9:N0}  TBRescr: {numTBRescored,9:N0}  "
                                + $"UnintendedBlund: {numUnintendedBlunders,9:N0}  NoiseBlund: {numNoiseBlunders,9:N0}");
        }
      }
    }


    private void RunGeneratorThread(ConcurrentQueue<string> files, Random rand)
    {
      while (true)
      {
        // Check if we have already written as many positions as requested.
        if (writer.NumPositionsWritten >= Options.NumPositionsTotal)
        {
          writer.Shutdown();
          return;
        }

        // Get next available file.
        if (!files.TryDequeue(out string fn))
        {
          throw new Exception("No input files available to process (are you using too many reader threads?).");
        }

        // Actually process all positions in file.
        DoProcessFN(fn);

        // Replace the file in the set of files to process (at the end of the queue).
        files.Enqueue(fn);
      }
    }


    void DoProcessFN(string fn)
    {
      try
      {
        Read(fn);
      }
      catch (Exception exc)
      {
        Console.WriteLine("Failure " + fn + " " + exc);
      }
    }

    [ThreadStatic]
    static TrainingPositionGeneratorGameRescorer gameAnalyzer;

    long numTBLookup = 0;
    long numTBFound = 0;
    long numTBRescored = 0;

    long numGamesProcessed;
    long numPositionsProcessed;
    long numFRCGamesSkipped;
    long numUnintendedBlunders;
    long numNoiseBlunders;

    int exceptionCount = 0;

    void Read(string fn)
    {
      // Every thread uses a random "skip modulus" which 
      // determines the positions that are skipped 
      // or actually emitted (if the sequence number of the scanned position
      // has remainder upon dividing by the SkipCount).
      int thisSkipModulus = 0;

      void ResetSkipModulus() => thisSkipModulus = (int)(DateTime.Now.Ticks % Options.PositionSkipCount);

      int numWrittenThisFile = 0;
      int numPosScannedThisFile = 0;

      // Note that we set filterOutFRCGames to false here, so we can see and count them,
      // but later in this method we filter them out.
      var reader = EncodedTrainingPositionReader.EnumerateGames(fn, s => true, filterOutFRCGames: false);

      Dictionary<ulong, int> positionUsedCountsByHash = new();

      int numGamesReadThisThread = 0;

      // To enhance random sampling, skip each thread skips a random number
      // of games from beginning of each file.
      int numGamesToSkipAtBeginningOfFile = Random.Shared.Next(500);

      try
      {
        foreach (EncodedTrainingPositionGame game in reader)
        {
          if (writer.NumPositionsWritten >= Options.NumPositionsTotal)
          {
            return;
          }

          numGamesReadThisThread++;
          if (numGamesReadThisThread < numGamesToSkipAtBeginningOfFile)
          {
            continue;
          }

          Interlocked.Increment(ref numGamesProcessed);

          // Always skip FRC games which could produce castling moves not understood by Ceres.
          if (game.IsFRCGame)
          {
            Interlocked.Increment(ref numFRCGamesSkipped);
            continue;
          }

          Interlocked.Add(ref numPosScanned, game.NumPositions);

          if (gameAnalyzer == null)
          {
            gameAnalyzer = new TrainingPositionGeneratorGameRescorer(Options.DeblunderThreshold, Options.EmitPlySinceLastMovePerSquare);
          }

          // Set up the game to analyze and run analysis so that
          // every position in the game will be annotated.
          gameAnalyzer.SetGame(game);
          gameAnalyzer.CalcBlundersAndTablebaseLookups(eval);
          gameAnalyzer.CalcTrainWDL(Options.Deblunder, Options.RescoreWithTablebase);

          // Update statistics.
          Interlocked.Add(ref numTBLookup, gameAnalyzer.numTBLookup);
          Interlocked.Add(ref numTBFound, gameAnalyzer.numTBFound);
          Interlocked.Add(ref numTBRescored, gameAnalyzer.numTBRescored);
          Interlocked.Add(ref numUnintendedBlunders, gameAnalyzer.numUnintendedBlunders);
          Interlocked.Add(ref numNoiseBlunders, gameAnalyzer.numNoiseBlunders);

          if (Options.Verbose)
          {
            gameAnalyzer.Dump();
          }


          for (int i = 0; i < game.NumPositions; i++)
          {
            numPosScannedThisFile++;

            if (numWrittenThisFile % 500 == 0)
            {
              // Enhance randomness by periodically resetting skip modulus
              ResetSkipModulus();
            }

            // Exit if this does not match our skip modulus.
            // However if the prior position at our skip modulus was
            // filtered out then keep sequentially looking for next non-filtered position.
            bool isModulusMatch = numPosScannedThisFile % Options.PositionSkipCount == thisSkipModulus;

            if (!isModulusMatch)
            {
              Interlocked.Increment(ref NumSkippedDueToModulus);
              continue;
            }

            if (gameAnalyzer != null)
            {
              if (gameAnalyzer.SHOULD_REJECT_POSITION[i])
              {
                continue;
              }
            }

            // Extract the position from the raw data.
            ref readonly EncodedPositionWithHistory thisGamePos = ref gameAnalyzer.PositionRef(i);
            Position thisPosition = thisGamePos.FinalPosition;

            // Possibly skip this position if it has already been written too many times.
            if (Options.PositionMaxFraction < 1)
            {
              ulong thisPositionHash = thisPosition.CalcZobristHash(PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98, false);
              positionUsedCountsByHash.TryGetValue(thisPositionHash, out int timesAlreadyUsed);
              float fractionOfTotalAlreadyUsed = (float)timesAlreadyUsed / numWrittenThisFile;
              if (fractionOfTotalAlreadyUsed >= Options.PositionMaxFraction)
              {
                Interlocked.Increment(ref numDuplicatesSkipped);
                continue;
              }
              else
              {
                positionUsedCountsByHash[thisPositionHash] = timesAlreadyUsed + 1;
              }
            }

            // Check the position filter to see if this should be accepted.
            if (Options.AcceptRejectAnnotater != null && !Options.AcceptRejectAnnotater(game, i, in thisPosition))
            {
              Interlocked.Increment(ref NumSkippedDueToPositionFilter);
              continue;
            }

            EncodedPositionEvalMiscInfoV6 thisTrainInfo = thisGamePos.MiscInfo.InfoTraining;

            if (game.Version != 6)
            {
              throw new Exception("TPGGenerator only supports version 6 training data, saw version " + game.Version);
            }

            // Do quick check and skip if BestQ or BestD are missing.
            bool missingBestInfo = float.IsNaN(thisTrainInfo.BestQ + thisTrainInfo.BestD);
            if (missingBestInfo)
            {
              Interlocked.Increment(ref NumSkippedDueToBadBestInfo);
              continue;
            }

            // Make sure this is a supported format and the record looks valid.
            EncodedTrainingPosition.ValidateIntegrity(game.InputFormat, game.Version,
                                                      thisGamePos, game.PolicyAtIndex(i),
                                                      "Data ingestion validation failure: " + fn);

            // Fill in missing (empty) history planes if requested.
            if (Options.FillInHistoryPlanes)
            {
              thisGamePos.FillInEmptyPlanes();
            }

#if NOT
            // Possibly annotate with tablebase entry.
            bool wasTBRescored = false;
            if (Options.RescoreWithTablebase)
            {
              // TODO: much more work here needed. Must look back into the game and rescore prior moves, etc.
              //       also many nuances in rescorer branch.
              // https://github.com/Tilps/lc0/blob/rescore_tb/src/selfplay/loop.cc#L204
              throw new NotImplementedException();

              //wasTBRescored = TablebasePositionRescoredWL(in thisPosition, gamePositionsBuffer, i);
            }
#endif

            // Accept this position.
            long indexThisPosUsed = Interlocked.Increment(ref numPosSentToWriter);
            numWrittenThisFile++;

            // Rotate written positions thru all sets to enhance diversity of positions within any single file.
            int setNum = (int)(indexThisPosUsed % Options.NumConcurrentSets);

#if NOT
            // Determine prior move
            EncodedMove? em = default;
            if (i > 0)
            {
              em = EncodedMove.FromNeuralNetIndex(gamePositionsBuffer[i - 1].PositionWithBoards.MiscInfo.InfoTraining.PlayedIndex);
              Console.WriteLine("playedmove1 " + thisPosition + " " + em + " " + em.Value.ToSquare.Flipped);
            }
#endif

            TrainingPositionWriterNonPolicyTargetInfo target = new();
            EncodedPositionEvalMiscInfoV6 infoTraining = game.PositionTrainingInfoAtIndex(i);
            target.ResultWDL = gameAnalyzer.newResultWDL[i];
            target.BestWDL = infoTraining.BestWDL;
            target.IntermediateWDL = gameAnalyzer.intermediateBestWDL[i];
            target.MLH = TPGRecordEncoding.MLHEncoded(infoTraining.PliesLeft);
            target.DeltaQVersusV = infoTraining.Uncertainty;
            target.DeltaQForwardAbs = gameAnalyzer.deltaQIntermediateBestWDL[i];
            target.Source = gameAnalyzer.targetSourceInfo[i];
            TrainingPositionWriterNonPolicyTargetInfo targetInfo = target;

            // TODO: avoid calling PositionAdIndex here
            EncodedTrainingPosition saveTrainingPos = new EncodedTrainingPosition(game.Version, game.InputFormat,
                                                                                  game.PositionAtIndex(i), game.PolicyAtIndex(i));
            const bool EMIT_MOVES = true;
            writer.Write(saveTrainingPos, in targetInfo, i, gameAnalyzer.lastMoveIndexBySquare?[i], Options.MinProbabilityForLegalMove, setNum, EMIT_MOVES);
          }
        }
      }
      catch (Exception exc)
      {
        Console.WriteLine("Exception processing TAR file, skipping partially: " + fn + " ");
        Console.WriteLine(exc + " " + exc.StackTrace);
        Console.WriteLine();

        if (exceptionCount++ > 500)
        {
          Console.WriteLine("Too many exceptions, aborting.");
          Environment.Exit(-1);
        }
      }

      bool shouldWrite = Options.TargetFileNameBase != null || numPosSentToWriter - numPosSentToWriterLastWriteLine >= INTERVAL_POSITIONS_WRITE_STATUS;
      if (shouldWrite)
      {
        // Write summary information.
        numPosSentToWriterLastWriteLine = numPosSentToWriter;
        float pctDone = 100.0f * (writer.NumPositionsWritten / (float)Options.NumPositionsTotal);
        lock (consoleOutputLock)
        {
          Console.WriteLine($"TPGRWITER : {pctDone,6:F2}%,  "
                          + $"{PosGeneratedPerSec,6:F0}/sec,  "
                          + $"scan: {numPosScanned,10:N0}  use: {numPosSentToWriter,9:N0}  skip_dups: {numDuplicatesSkipped,9:N0}  "
                          + $"FRC_reject: {numFRCGamesSkipped,9:N0} "
                          + $"reject_pre: {NumSkippedDueToPositionFilter,9:N0} "
                          + $"reject_post: {writer.numPositionsRejectedByPostprocessor,9:N0} "
                          + $"TBLook: {numTBLookup,9:N0}  TBFound: {numTBFound,9:N0}  TB_rescr: {numTBRescored,9:N0}  "
                          + $"err_blund: {numUnintendedBlunders,9:N0}  noise_blund: {numNoiseBlunders,9:N0}  {fn}");
        }

      }
    }

    const long INTERVAL_POSITIONS_WRITE_STATUS = 1_000_000;
    long numPosSentToWriterLastWriteLine = -INTERVAL_POSITIONS_WRITE_STATUS;


    static bool PostprocessFocusOnly(Position position,
                                     NNEvaluatorResult nnEvalResult,
                                     in EncodedTrainingPosition trainingPosition)
    {
      // Always accept 20% unconditionally
      if (position.CalcZobristHash(PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98) % 5 == 0) return true;

      float evalV = nnEvalResult.V;
      float searchQ = trainingPosition.PositionWithBoards.MiscInfo.InfoTraining.ResultQ;
      float vErrAbs = MathF.Abs(evalV - searchQ);

      ref readonly CompressedPolicyVector evalPolicy = ref nnEvalResult.Policy;
      var evalProbs = evalPolicy.ProbabilitySummary().ToArray();
      ref readonly EncodedPolicyVector searchPolicy = ref trainingPosition.Policies;
      float sumDiff = 0;
      foreach (var prob in evalProbs)
      {
        float probOther = searchPolicy[prob.Move.IndexNeuralNet].probability;
        sumDiff += MathF.Abs(prob.Probability - probOther);
      }

      bool isBlunder = vErrAbs > 0.20f || sumDiff > 1.3f;
      return isBlunder;
    }
  }
}
