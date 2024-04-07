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
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;
using Ceres.Base.Math.Random;
using System.Diagnostics;
using Ceres.Base.Math;


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

    bool PositionAtIndexShouldBeProcessed(int i, EncodedTrainingPositionGame game, 
                                          bool includeCheckForPositionMaxFraction,
                                          Dictionary<ulong, int> positionUsedCountsByHash, 
                                          int numWrittenThisFile)
    {
      if (gameAnalyzer != null)
      {
        if (gameAnalyzer.SHOULD_REJECT_POSITION[i])
        {
          return false;
        }
      }

      // Extract the position from the raw data.
      ref readonly EncodedPositionWithHistory thisGamePos = ref gameAnalyzer.PositionRef(i);
      Position thisPosition = thisGamePos.FinalPosition;

      // Possibly skip this position if it has already been written too many times.
      if (includeCheckForPositionMaxFraction && Options.PositionMaxFraction < 1)
      {
        ulong thisPositionHash = thisPosition.CalcZobristHash(PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98, false);
        positionUsedCountsByHash.TryGetValue(thisPositionHash, out int timesAlreadyUsed);
        float fractionOfTotalAlreadyUsed = (float)timesAlreadyUsed / numWrittenThisFile;
        if (fractionOfTotalAlreadyUsed >= Options.PositionMaxFraction)
        {
          Interlocked.Increment(ref numDuplicatesSkipped);
          return false;
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
        return false;
      }

      return true;
    }


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

      Span<EncodedMove> policyMoves = stackalloc EncodedMove[CompressedPolicyVector.NUM_MOVE_SLOTS];
      Span<float> policyProbs = stackalloc float[CompressedPolicyVector.NUM_MOVE_SLOTS];

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
            gameAnalyzer = new TrainingPositionGeneratorGameRescorer(Options.DeblunderThreshold,
                                                                     Options.DeblunderUnintnededThreshold,
                                                                     Options.EmitPlySinceLastMovePerSquare);
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

            // Verify this position acceptable to process.
            bool okToProcess = PositionAtIndexShouldBeProcessed(i, game, true, positionUsedCountsByHash, numWrittenThisFile);
            if (!okToProcess)
            {
              continue;
            }

            // If we are emitting mutiple boards as a block,
            // there are additional restrictions on which positions are suitable to emit.
            if (Options.NumRelatedPositionsPerBlock > 1)
            {
              if (Options.NumRelatedPositionsPerBlock != 4)
              {
                throw new NotImplementedException("Only 4 related positions per block supported.");
              }

              if (i >= game.NumPositions - 2)
              {
                // If doing in pairs, can only accept this position if there is another position following.
                continue;
              }

              // Check the next 2 moves and abort unless the are both:
              //   -- acceptable on a standalone bases, and
              //   -- not too far off the optimal play line
              double PlayedMoveSuboptimalityAtIndex(int i) => game.PositionAtIndex(i).MiscInfo.InfoTraining.QSuboptimality;
              const float MAX_PRIOR_MOVE_SUBOPTIMALITY = 0.02f;
              if (!PositionAtIndexShouldBeProcessed(i + 1, game, false, positionUsedCountsByHash, numWrittenThisFile)
                && PlayedMoveSuboptimalityAtIndex(i) < MAX_PRIOR_MOVE_SUBOPTIMALITY)
              {
                // Continue since we want to also use the single-next position but it is not acceptable.
                continue;
              }
              if (!PositionAtIndexShouldBeProcessed(i + 2, game, false, positionUsedCountsByHash, numWrittenThisFile)
                && PlayedMoveSuboptimalityAtIndex(i + 1) < MAX_PRIOR_MOVE_SUBOPTIMALITY)

              {
                // Continue since we want to also use the double-next position
                // but it is either not acceptable standalone or too far off the optimal play line
                // (suboptimal move choice by prior position).
                continue;
              }
            }

            // Update counters to reflect this position or block of positions to be written.
            long indexThisPosUsed = Interlocked.Add(ref numPosSentToWriter, Options.NumRelatedPositionsPerBlock);
            numWrittenThisFile += Options.NumRelatedPositionsPerBlock;
            long numBlocksWrittenThisFile = indexThisPosUsed / Options.NumRelatedPositionsPerBlock;

            // Rotate written positions thru all sets to enhance diversity of positions within any single file.
            int setNum = (int)(numBlocksWrittenThisFile % Options.NumConcurrentSets);

            const bool EMIT_MOVES = true;

            if (Options.NumRelatedPositionsPerBlock == 1)
            {
              // Simple case of just a single board.
              var pendingItem = PreparePosition(setNum, fn, game, i);
              writer.Write(setNum, Options.MinProbabilityForLegalMove, EMIT_MOVES, (pendingItem, true));
            }
            else
            {
              // First 3 slots are easy, just use the next 3 positions.
              var item1 = PreparePosition(setNum, fn, game, i);
              var item2 = PreparePosition(setNum, fn, game, i + 1);
              var item3 = PreparePosition(setNum, fn, game, i + 2);

              // Now pick the 4th position randomly as one of the possible continuations
              // from the root position (but not the one actually chosen).
              EncodedTrainingPosition thisPos = item1.record;

              // Extract Spans for the policy moves and probabilities from the root.
              int policyLen = thisPos.Policies.ExtractIntoSpans(policyMoves, policyProbs);

              MGPosition startMGPos = game.PositionAtIndex(i).FinalPosition.ToMGPosition;

              // Helper method which creates new training data for position after specified move.
              (EncodedTrainingPosition, TrainingPositionWriterNonPolicyTargetInfo, int, short[])
                MakeForDrawIndex(EncodedMove encodedMove)
              {
                //                short moveIndex = policyMoves[drawIndex].IndexNeuralNet;
                MGMove move3 = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(encodedMove, in startMGPos);
                TrainingPositionWriterNonPolicyTargetInfo target3 = default;
                target3.PolicyIndexInParent = (short)ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(move3).IndexNeuralNet;

                EncodedTrainingPosition pos3 = TrainingPositionAfterMove(game.TrainingPosition(i), move3);
                return (pos3, target3, -1, null);
              }

              (EncodedTrainingPosition, TrainingPositionWriterNonPolicyTargetInfo, int, short[]) item4;
              if (policyLen == 1)
              {
                // Only one move choice, must use this a second time.
                item4 = item2;
              }
              else
              {
                // Clear the probability of the already used move to 0 so it will not be chosen.
                EncodedMove moveMadeInGame = EncodedMove.FromNeuralNetIndex(thisPos.PositionWithBoards.MiscInfo.InfoTraining.PlayedIndex);
                bool found = false;
                for (int ix=0;ix<policyLen; ix++) 
                {
                  if (policyMoves[ix] == moveMadeInGame)
                  {
                    policyProbs[ix] = 0;
                    found = true;
                    break;
                  }
                }
                Debug.Assert(found);

                StatUtils.Normalize(policyProbs);

                // The other 3 slots will always have the top policy move as the action move target.
                // This top policy move will have a value close to the output of the value head.
                // This creates a strong bias high bias in the action targets seen.
                //
                // To partly counteract this, make sure the 4th board is drawn from moves which 
                // are well distributed across all moves (including bad one).
                // However we would also like to give somewhat more weight to the "almost best" policy moves
                // because they of more importance in gameplay.
                // To achieve a balance, we therefore assign half the weight based on policy
                // and half from a uniform distribution, insuring poor moves get considerable representation.
                //
                // Doubtless the net will nevertheless have an optimistic bias in the action outputs,
                // but this may not be so bad since search needs some optimism bias to not totally squelch exploration.
                float uniformWeight = 1.0f / policyLen;
                for (int ip=0;ip<policyLen;ip++)
                {
                  policyProbs[ip] = 0.5f * policyProbs[ip] 
                                  + 0.5f * uniformWeight;
                } 

                int tryDrawIndex = ThompsonSampling.Draw(policyProbs, policyLen);
                item4 = MakeForDrawIndex(policyMoves[tryDrawIndex]);
              }
            
              
              writer.Write(setNum, Options.MinProbabilityForLegalMove, EMIT_MOVES, 
                           (item1, true), (item2, true), (item3, true), (item4, false));

            }
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

      PossiblyWriteStatusMessage(fn);
    }


    private (EncodedTrainingPosition record, TrainingPositionWriterNonPolicyTargetInfo targetInfo, int indexMoveInGame, short[] indexLastMoveBySquares)
      PreparePosition(int setNum, string fn, EncodedTrainingPositionGame game, int i)
    {
      // Extract the position from the raw data.
      ref readonly EncodedPositionWithHistory thisGamePos = ref gameAnalyzer.PositionRef(i);
      Position thisPosition = thisGamePos.FinalPosition;
      EncodedPositionEvalMiscInfoV6 thisTrainInfo = thisGamePos.MiscInfo.InfoTraining;

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
      target.ResultNonDeblunderedWDL = infoTraining.ResultWDL;
      target.ResultDeblunderedWDL = gameAnalyzer.newResultWDL[i];
      target.BestWDL = infoTraining.BestWDL;
      target.IntermediateWDL = gameAnalyzer.intermediateBestWDL[i];
      target.MLH = TPGRecordEncoding.MLHEncoded(infoTraining.PliesLeft);
      target.DeltaQVersusV = infoTraining.Uncertainty;

      // Emit prior position Win/Loss if requested in options
      // and this move was not a big blunder (which makes prior evaluation not informative).
      (float w, float d, float l) = (0, 0, 0);
      const float SUBOPTIMALITY_THRESHOLD = 0.10f;
      if (Options.EmitPriorMoveWinLoss
       && infoTraining.QSuboptimality < SUBOPTIMALITY_THRESHOLD)
      {
        (w, d, l) = CalculatePriorPositionWDL(game, i);
      }

      // Record this prior position win/loss, reversing to reflect our perspective.
      target.PriorPositionWinP = l;
      target.PriorPositionDrawP = d;
      target.PriorPositionLossP = w;

      target.PolicyIndexInParent = i == 0 ? (short)-1
                                          : (short)gameAnalyzer.PositionRef(i - 1).MiscInfo.InfoTraining.PlayedIndex; 

      target.DeltaQForwardAbs = gameAnalyzer.deltaQIntermediateBestWDL[i];
      target.Source = gameAnalyzer.targetSourceInfo[i];

      target.ForwardSumPositiveBlunders = gameAnalyzer.forwardSumPositiveBlunders[i];
      target.ForwardSumNegativeBlunders = gameAnalyzer.forwardSumNegativeBlunders[i];

      target.ForwardMinQDeviation = gameAnalyzer.forwardMinQDeviation[i];
      target.ForwardMaxQDeviation = gameAnalyzer.forwardMaxQDeviation[i];

      TrainingPositionWriterNonPolicyTargetInfo targetInfo = target;

      // TODO: avoid calling PositionAdIndex here
      EncodedTrainingPosition saveTrainingPos = new EncodedTrainingPosition(game.Version, game.InputFormat,
                                                                            game.PositionAtIndex(i), game.PolicyAtIndex(i));
      return (saveTrainingPos, targetInfo, i, gameAnalyzer.lastMoveIndexBySquare?[i]);
    }



    private void PossiblyWriteStatusMessage(string fn)
    {
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


    /// <summary>
    /// 
    /// Note that in cases below where we fail to have good data for the prior position,
    /// we set everything to zero so the net sees 0 for both Win and Loss.
    /// The net should learn to understand this means "no data available."
    /// 
    /// This also makes it  possible to use the network in situations where 
    /// prior evaluation not available (e.g. in a suite test position).
    ///
    /// </summary>
    /// <param name="game"></param>
    /// <param name="i"></param>
    /// <returns></returns>
    private (float w, float d, float l) CalculatePriorPositionWDL(EncodedTrainingPositionGame game, int i)
    {
      float w, d, l;

      // Can't look at prior position if this is the first.
      if (i == 0)
      {
        return (0, 0, 0);
      }

      // Get the training information for the prior position.
      EncodedPositionEvalMiscInfoV6 infoTraining = game.PositionTrainingInfoAtIndex(i - 1);

      (w, d, l) = infoTraining.OriginalWDL;

      // Under two conditins we don't valid data here.
      //   - the OriginalWDL is NaN (happens rarely in the data)
      //   - this was a big blunder so the prior value score is not very relevant
      if (float.IsNaN(w + d + l))
      {
        return (0, 0, 0);
      }     

      // Return values, reversed to convert to our perspetive perspective
      return (l, d, w);
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



    /// <summary>
    /// 
    /// NOTE: patterned after code in LC0TrainingPosGeneratorFromSingleNNEval.cs
    ///       probably move this back there
    /// </summary>
    /// <param name="startPos"></param>
    /// <param name="moveToPlay"></param>
    /// <returns></returns>
    static EncodedTrainingPosition TrainingPositionAfterMove(in EncodedTrainingPosition startPos, MGMove moveToPlay)
    {
      // Get current position with history and the played move.
      PositionWithHistory currentPos = startPos.ToPositionWithHistory(8);
      MGPosition thisPos = currentPos.FinalPosition.ToMGPosition;

      MGPosition nextPos = thisPos;
      nextPos.MakeMove(moveToPlay);

      PositionWithHistory nextPosition = new PositionWithHistory(currentPos);
      nextPosition.AppendPosition(nextPos, moveToPlay);

      EncodedPositionWithHistory newPosHistory = default;
      newPosHistory.SetFromSequentialPositions(nextPosition.Positions, false); // this also takes care of the misc info

      EncodedPositionEvalMiscInfoV6 trainingMiscInfo = default;
#if NOT
      new
        (
        invarianceInfo: startPos.PositionWithBoards.MiscInfo.InfoTraining.InvarianceInfo, depResult: default,
        rootQ: 0, bestQ: 0, rootD: 0, bestD: 0,
        rootM: 0, bestM: 0, pliesLeft: 0,
        resultQ: 0, resultD: 0,
        playedQ: 0, playedD: 0, playedM: 0,
        originalQ: 0, originalD: 0, originalM: 0,
        numVisits: 1,
        playedIndex: (short)bestMoveIndex, bestIndex: (short)bestMoveIndex,
        unused1: default, unused2: default);
#endif

      EncodedTrainingPositionMiscInfo miscInfoAll = new(newPosHistory.MiscInfo.InfoPosition, trainingMiscInfo);
      newPosHistory.SetMiscInfo(miscInfoAll);

      EncodedPolicyVector epv = default;
      epv.InitilializeAllNegativeOne();

      return new EncodedTrainingPosition(startPos.Version, startPos.InputFormat, newPosHistory, epv);
    }


  }
}
