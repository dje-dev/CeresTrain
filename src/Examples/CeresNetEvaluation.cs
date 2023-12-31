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

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.MCTS.Params;
using Ceres.Features.Players;
using Ceres.Features.Tournaments;
using Ceres.Features.GameEngines;

using CeresTrain.NNEvaluators;
using CeresTrain.PositionGenerators;
using CeresTrain.Trainer;
using Ceres.Chess.LC0.Batches;
using CeresTrain.TPG;
using CeresTrain.TrainingDataGenerator;
using System.Linq;
using Spectre.Console;
using Ceres.Chess.Data.Nets;
using Ceres.Chess.Positions;
using Ceres.Base.Benchmarking;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Zstandard.Net;
using Ceres.Base.DataType;

#endregion 

namespace CeresTrain.Examples
{
  /// <summary>
  /// Set of static methods to facilitate running various tests of Ceres neural networks.
  /// </summary>
  public static class CeresNetEvaluation
  {

    /// <summary>
    /// Generates a set of TPG files from random endgame positions having specified set of pieces.
    /// </summary>
    /// <param name="piecesString"></param>
    /// <param name="numPositions"></param>
    /// <param name="outputDirectory"></param>
    public static void GenerateTPGFilesFromRandomTablebasePositions(string piecesString, long numPositions, string outputDirectory)
    {
      if (!Directory.Exists(outputDirectory))
      {
        Directory.CreateDirectory(outputDirectory);
      }

      // Construct a PieceList from the pieces string (verifies validity).
      PieceList pieceList = new(piecesString);

      int NUM_THREADS = 4 + (int)MathF.Min(16, Environment.ProcessorCount / 2);
      const int BATCH_SIZE = 1024;
      long NUM_BATCHES_TO_WRITE = numPositions / (NUM_THREADS * BATCH_SIZE);
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"GenerateTPGFileFromRandomTablebasePositions positions {numPositions} "
                                                     + $"of {piecesString} with {NUM_THREADS} threads to {outputDirectory}");

      // Launch and wait for parallel threads, each writing to separate file.
      DateTime startTime = DateTime.Now;
      const bool SUCCEED_IF_INCOMPLETE_DTZ_INFORMATION = true;
      using (new TimingBlock("GenerateTPGFilesFromRandomTablebasePositions"))
      {
        List<Task> tasks = new();
        Enumerable.Range(0, NUM_THREADS).ToList().ForEach(i =>
        {
          tasks.Add(Task.Run(() => GenerateTPGFileFromRandomTablebasePositions(SUCCEED_IF_INCOMPLETE_DTZ_INFORMATION, outputDirectory, piecesString, i, BATCH_SIZE, NUM_BATCHES_TO_WRITE)));
        });
        Task.WaitAll(tasks.ToArray());
      }

      DateTime endTime = DateTime.Now;
      int nps = (int)((float)(NUM_THREADS * BATCH_SIZE * NUM_BATCHES_TO_WRITE) / (endTime - startTime).TotalSeconds);
      Console.WriteLine();
      Console.WriteLine($"Done: {nps:N0} positions per second generation to TPG files.");
    }


    /// <summary>
    /// Thread worker method for writing a batch of TPG records to a file.
    /// </summary>
    /// <param name="succeedIfIncompleteDTZInformation"></param>
    /// <param name="outputDirectory"></param>
    /// <param name="piecesString"></param>
    /// <param name="filnameIndex"></param>
    /// <param name="batchSize"></param>
    /// <param name="numBatches"></param>
    static void GenerateTPGFileFromRandomTablebasePositions(bool succeedIfIncompleteDTZInformation, string outputDirectory, string piecesString, int filnameIndex, int batchSize, long numBatches)
    {
      PositionGeneratorRandomFromPieces generator = new PositionGeneratorRandomFromPieces(piecesString);
      TablebaseTPGBatchGenerator tpgGenerator = new(generator.ID, generator.GeneratePosition, succeedIfIncompleteDTZInformation, batchSize);

      string outFN = Path.Combine(outputDirectory, @$"{FileUtils.FileNameSanitized(generator.ID)}_{filnameIndex}.dat.zst");

      const int COMPRESSION_LEVEL = 10;
      using FileStream fs = new FileStream(outFN, FileMode.Create, FileAccess.Write);
      using ZstandardStream zs = new ZstandardStream(fs, COMPRESSION_LEVEL);

      long numBatchesWritten = 0;
      foreach (TPGRecord[] batch in tpgGenerator.Enumerator())
      {
        // Write compressed to the windows.
        StreamUtils.WriteSpanToStream<TPGRecord>(zs, batch);

        // Stop once enough records written.
        if (++numBatchesWritten == numBatches)
        {
          break;
        }
      }

      Console.WriteLine("Done " + outFN);
      tpgGenerator.Shutdown();
    }


    /// <summary>
    /// Method that returns a neural network evaluator for a given trained network on disk. 
    /// </summary>
    /// <param name="netDef"></param>
    /// <param name="execConfig"></param>
    /// <param name="netFN"></param>
    /// <param name="useBestValueRepetitionHeuristic"></param>
    /// <returns></returns>
    public static NNEvaluatorTorchsharp GetNNEvaluator(ICeresNeuralNetDef netDef, in ConfigNetExecution execConfig,
                                                       string netFN, bool useBestValueRepetitionHeuristic)
    {
      NNEvaluatorTorchsharp evaluator = new(netDef, execConfig with { SaveNetwork1FileName = netFN }, false);
      evaluator.UseBestValueMoveUseRepetitionHeuristic = useBestValueRepetitionHeuristic;
      return evaluator;
    }


    /// <summary>
    /// Run a match between the specified trained network and the LC0 reference network,
    /// using either the pure value or pure policy head outputs.
    /// </summary>
    /// <param name="netFN"></param>
    /// <param name="opponentNetID"></param>
    /// <param name="ceresDeviceSpec"></param>
    /// <param name="posGenerator"></param>
    /// <param name="searchLimit"></param>
    /// <param name="numGamePairs"></param>
    /// <param name="verbose"></param>
    public static TournamentResultStats RunTournament(ICeresNeuralNetDef netDef, in ConfigNetExecution execConfig,
                                                      string netFN, string opponentNetID, string ceresDeviceSpec, 
                                                      PositionGenerator posGenerator,
                                                      SearchLimit searchLimit, int numGamePairs = 50, bool verbose = false, 
                                                      bool opponentTablebasesEnabled = false)
    {
      EnginePlayerDef player1;
      EnginePlayerDef player2;

      if (netFN != null)
      {
        // Playing Ceres net versus LC0 net (with or without tablebases, as determined by opponentTablebasesEnabled).
        InstallCustomEvaluator(netDef, in execConfig, netFN, posGenerator, opponentNetID, ceresDeviceSpec);
        player1 = GetPlayerDef("Ceres1", "CUSTOM1", ceresDeviceSpec, searchLimit, false);
        player2 = GetPlayerDef("Ceres2", opponentNetID, ceresDeviceSpec, searchLimit, opponentTablebasesEnabled);
      }
      else
      {
        // Playing LC0 net (without tablebases) against itself (with tablebases)
        player1 = GetPlayerDef(opponentNetID, opponentNetID, ceresDeviceSpec, searchLimit, false);
        player2 = GetPlayerDef(opponentNetID + "_TB", opponentNetID, ceresDeviceSpec, searchLimit, opponentTablebasesEnabled);
      }

      TournamentDef def = new("TOURN", player1, player2);
      def.ShowGameMoves = verbose;
      def.UseTablebasesForAdjudication = false;
      def.NumGamePairs = numGamePairs;
      def.OpeningsFileName = posGenerator.GeneratedTestEPDFile(def.NumGamePairs.Value);

      // Run the tournament.
      return new TournamentManager(def).RunTournament();
    }


    /// <summary>
    /// Test the accuracy of the specified trained network on a set of random endgame positions,
    /// listing the positions, indicating where the value and/or policy heads deviate from tablebase ground truth.
    /// </summary>
    /// <param name="generator">generator for positions of the type desired to be analyzed</param>
    /// <param name="sourceEPDOrPGN">optional PGN file from which positions are sourced (instead of random)</param>
    /// <param name="evaluatorPrimary">file containing saved CeresTrain neural network</param>
    /// <param name="evaluatorCompare">optional comparison evaluator</param>
    /// <param name="trainingResult">optional TrainingResultSummary summarizing training results for the net</param>
    /// <param name="numPos">number of positions to test</param>
    /// <param name="verbose">if details of each tested position should be dumped to Console</param>
    public static (float accuracyValue, float accuracyPolicy) TestAccuracyOnPositions(PositionGenerator generator, string sourceEPDOrPGN,
                                                                                      NNEvaluator evaluatorPrimary, NNEvaluator evaluatorCompare = null,
                                                                                      TrainingResultSummary trainingResult = default,
                                                                                      int numPos = 50, bool verbose = false)
    {
      bool FILL_IN_HISTORY = !evaluatorPrimary.UseBestValueMoveUseRepetitionHeuristic; // Some LC0 nets seem to require; Ceres nets will override to ignore

      ISyzygyEvaluatorEngine tbEvaluator = ISyzygyEvaluatorEngine.DefaultEngine;

      int numValueCorrectPrimary = 0;
      int numPolicyCorrectPrimary = 0;
      int numValueCorrectCompare = 0;
      int numPolicyCorrectCompare = 0;

      IEnumerable<PositionWithHistory> positionSourceEnum = sourceEPDOrPGN == null
                                            ? generator.AsPositionWithHistoryEnumerable()
                                            : PositionsWithHistory.FromEPDOrPGNFile(sourceEPDOrPGN, numPos, p => generator.PositionMatches(in p)).AsEnumerable();
      IEnumerator<PositionWithHistory> posEnumerator = positionSourceEnum.AsEnumerable().GetEnumerator();

      for (int i = 0; i < numPos; i++)
      {
        if (!posEnumerator.MoveNext())
        {
          throw new Exception($"Insufficient positions matching {generator} found in file {sourceEPDOrPGN}");
        }

        PositionWithHistory pos = posEnumerator.Current;

        NNEvaluatorResult evalResult = evaluatorPrimary.Evaluate(pos, FILL_IN_HISTORY);
        NNEvaluatorResult resultCompar = evaluatorCompare == null ? default : evaluatorCompare.Evaluate(pos, true, extraInputs:NNEvaluator.InputTypes.Positions);

        int gameResultTablebase = tbEvaluator.ProbeWDLAsV(pos.FinalPosition);
        bool netValueCorrect = gameResultTablebase == evalResult.MostProbableGameResult;

        bool netTopMoveInBestCategory = tbEvaluator.MoveIsInOptimalCategoryForPosition(pos.FinalPosition, evalResult.Policy.TopMove(pos.FinalPosition), true);
        bool netTopMoveInBestCategoryCompare = evaluatorCompare == null ? false : tbEvaluator.MoveIsInOptimalCategoryForPosition(pos.FinalPosition, resultCompar.Policy.TopMove(pos.FinalPosition), true);

        bool compareOK = resultCompar.MostProbableGameResult == gameResultTablebase;
        if (verbose)// && (!netValueCorrect || !compareOK))
        {
          Console.Write($"TB= {gameResultTablebase,2:n0}  Primary=");
          WriteFloatRedIfNegative(evalResult.V, netValueCorrect);
          Console.Write("  Comp=");
          WriteFloatRedIfNegative(evaluatorCompare == null ? float.NaN : resultCompar.V, compareOK);

          Console.WriteLine($" {evalResult.Policy,-115}   {pos.FinalPosition.FEN}");
        }

        numValueCorrectPrimary += netValueCorrect ? 1 : 0;
        numValueCorrectCompare += compareOK ? 1 : 0;
        numPolicyCorrectPrimary += netTopMoveInBestCategory ? 1 : 0;
        numPolicyCorrectCompare += netTopMoveInBestCategoryCompare ? 1 : 0;
      }

      Console.WriteLine();
      if (trainingResult != default)
      {
        Console.WriteLine($"Trained {trainingResult.NumTrainingPositions:N0} steps in {trainingResult.TrainingTime} with {trainingResult.NumParameters} parameters");
        Console.WriteLine($"  Loss: {trainingResult.LossSummary.TotalLoss:F2}");
        Console.WriteLine($"  Value Loss/Accuracy       : {trainingResult.LossSummary.ValueLoss:F2}  {100 * trainingResult.LossSummary.ValueAccuracy:F2}%");
        Console.WriteLine($"  Policy Loss/Accuracy      : {trainingResult.LossSummary.PolicyLoss:F2}  {100 * trainingResult.LossSummary.PolicyAccuracy:F2}%");
      }

      Console.WriteLine("Testing Evaluator " + evaluatorPrimary.ToString());
      Console.WriteLine($"Number of test positions : {numPos}  {generator.ID}  {evaluatorPrimary.EngineNetworkID}");

      float accuracyValue = (100.0f * ((float)numValueCorrectPrimary / numPos));
      Console.WriteLine($"  Accuracy value        : {accuracyValue:F2}%");
      if (evaluatorCompare != null)
      {
        Console.WriteLine($"  Accuracy value (comp) : {(100.0f * ((float)numValueCorrectCompare / numPos)):F2}% using {evaluatorCompare}");
      }

      float accuracyPolicy = (100.0f * ((float)numPolicyCorrectPrimary / numPos));
      Console.WriteLine();
      Console.WriteLine($"  Accuracy policy       : {accuracyPolicy:F2}%");
      if (evaluatorCompare != null)
      {
        Console.WriteLine($"  Accuracy policy (comp): {(100.0f * ((float)numPolicyCorrectCompare / numPos)):F2}% using {evaluatorCompare}");
      }

      Console.WriteLine();
      Console.WriteLine($"Inference speed: {NNEvaluatorBenchmark.EstNPS(evaluatorPrimary)}");
      Console.WriteLine();

      return (accuracyValue, accuracyPolicy);
    }


    /// <summary>
    /// Computes the value head accuracy of a specified evaluator on a random positions.
    /// </summary>
    /// <param name="netFN"></param>
    /// <param name="numPositionsToTest"></param>
    public static (float valueAccuracy, float policyAccuracy) TestNetValueAccuracy(NNEvaluator evaluator, string piecesStr, 
                                                                                  int numPositionsToTest = 4096, 
                                                                                  string epdOrPGNFileName = null,
                                                                                  bool verbose = false)
    {
      bool FILL_IN_HISTORY = !evaluator.UseBestValueMoveUseRepetitionHeuristic;

      ISyzygyEvaluatorEngine tbEvaluator = ISyzygyEvaluatorEngine.DefaultEngine;
      PositionGeneratorRandomFromPieces generator = new PositionGeneratorRandomFromPieces(piecesStr);

      IEnumerable<Position> positionSourceEnum = epdOrPGNFileName == null
                                            ? generator.AsPositionEnumerable()
                                            : PositionsWithHistory.FromEPDOrPGNFile(epdOrPGNFileName, numPositionsToTest, p => generator.PositionMatches(in p)).Select(s=>s.FinalPosition).AsEnumerable();
      IEnumerator<Position> posEnumerator = positionSourceEnum.AsEnumerable().GetEnumerator();

      int numCorrectValue = 0;
      int numCorrectPolicy = 0;
      int numTested = 0;

      int BATCH_SIZE = evaluator.MaxBatchSize;
      EncodedPositionBatchBuilder builder = new(BATCH_SIZE, evaluator.InputsRequired | NNEvaluator.InputTypes.Positions);
      Position[] batchPositions = new Position[BATCH_SIZE];

      while (numTested < numPositionsToTest)
      {
        int numThisBatch = Math.Min(numPositionsToTest - numTested, BATCH_SIZE);
        builder.ResetBatch();
        for (int i = 0; i < numThisBatch; i++)
        {
          if (!posEnumerator.MoveNext())
          {
            throw new Exception($"Insufficient positions found");
          }

          Position pos = posEnumerator.Current;
          batchPositions[i] = pos;
          builder.Add(in pos, FILL_IN_HISTORY);
        }

        EncodedPositionBatchFlat batch = builder.GetBatch();
        NNEvaluatorResult[] nnResults = evaluator.EvaluateBatch(batch);

        for (int i = 0; i < numThisBatch; i++)
        {
          bool policyCorrect = tbEvaluator.MoveIsInOptimalCategoryForPosition(in batchPositions[i], nnResults[i].Policy.TopMove(in batchPositions[i]), true);
          if (policyCorrect)
          {
            numCorrectPolicy++;
          }

          bool valueCorrect = tbEvaluator.ProbeWDLAsV(batchPositions[i]) == nnResults[i].MostProbableGameResult;
          if (valueCorrect)
          {
            numCorrectValue++;
          }

          if (verbose)
          {
            Console.WriteLine((valueCorrect ? " " : "v") + " " + (policyCorrect ? " " : "p") 
                            + $"  {batchPositions[i].FEN,-35}   {nnResults[i].V,8:F2} { nnResults[i].Policy}");
          }

        }
        numTested += numThisBatch;
      }

      if (verbose)
      {
        Console.WriteLine();
      }

      float valueAccuracy = (float)numCorrectValue / numTested;
      float policyAccuracy = (float)numCorrectPolicy / numTested;

      return (valueAccuracy, policyAccuracy);
    }


    /// <summary>
    /// Computes the value head accuracy of a specified network on a set of random positions.
    /// </summary>
    /// <param name="netFN"></param>
    /// <param name="numPositionsToTest"></param>
    public static void TestNetValueAccuracy(NNEvaluatorTorchsharp evaluator, string piecesStr, int numPositionsToTest = 4096)
    {
      ISyzygyEvaluatorEngine tbEvaluator = ISyzygyEvaluatorEngine.DefaultEngine;
      PositionGeneratorRandomFromPieces generator = new PositionGeneratorRandomFromPieces(piecesStr);

      int numCorrect = 0;
      int numTested = 0;

      while (numTested < numPositionsToTest)
      {
        // Get raw batch then reduce (if necessary) to max batch size supported by evaluator.
        const bool SUCCEED_IF_INCOMPLETE_DTZ_INFORMATION = true;
        TPGRecord[] batchRaw = new TablebaseTPGBatchGenerator(piecesStr, generator.GeneratePosition, SUCCEED_IF_INCOMPLETE_DTZ_INFORMATION).Enumerator().First();
        Span<TPGRecord> batch = new Span<TPGRecord>(batchRaw, 0, Math.Min(batchRaw.Length, evaluator.MaxBatchSize));

        IPositionEvaluationBatch nnResults = evaluator.Evaluate(batch);

        for (int i = 0; i < batch.Length; i++)
        {
          int gameResultTablebase = tbEvaluator.ProbeWDLAsV(batch[i].FinalPosition);

          evaluator.ExtractToNNEvaluatorResult(out NNEvaluatorResult nnResult, nnResults, i);
          //Console.WriteLine(gameResultTablebase + " " + Math.Round(nnResult.V) + " " + Math.Round(nnResults.GetV(i)));
          if (gameResultTablebase == nnResult.MostProbableGameResult)
          {
            numCorrect++;
          }
        }
        numTested += batch.Length;
      }

      Console.WriteLine("Value accuracy: " + (float)numCorrect / numTested);
    }


    /// <summary>
    /// Launch UCI loop with specified network to allow user to play against it from the command line.
    /// The Ceres engine is launched with tablebases disabled so the neural network is not bypassed.
    /// </summary> 
    /// <param name="netFN"></param>
    /// <param name="lc0NetToUseForUncoveredPositions"></param>
    /// <param name="ceresDeviceSpec"></param>
    public static void RunUCILoop(ICeresNeuralNetDef netDef, in ConfigNetExecution execConfig,
                                  string netFN, string lc0NetToUseForUncoveredPositions, 
                                  string ceresDeviceSpec, PositionGenerator posGenerator)
    {
      InstallCustomEvaluator(netDef, in execConfig, netFN, posGenerator, lc0NetToUseForUncoveredPositions, ceresDeviceSpec);
      Ceres.Program.LaunchUCI(["network=CUSTOM1"], (ParamsSearch search) => { search.EnableTablebases = false; });
    }





    /// <summary>
    /// Returns an EnginePlayerDef for use in a TournamentDef based on specified parameters.
    /// </summary>
    /// <param name="playerID"></param>
    /// <param name="netSpecString"></param>
    /// <param name="ceresDeviceSpec"></param>
    /// <param name="limit"></param>
    /// <param name="enableTablebases"></param>
    /// <returns></returns>
    static EnginePlayerDef GetPlayerDef(string playerID, string netSpecString, string ceresDeviceSpec, SearchLimit limit, bool enableTablebases = false)
    {
      NNEvaluatorDef evalDef = NNEvaluatorDef.FromSpecification(netSpecString, ceresDeviceSpec);
      GameEngineDefCeres engineDef = new(playerID, evalDef, default,
                                          new ParamsSearch() { EnableTablebases = enableTablebases },
                                          new ParamsSelect(), null, "CeresEG.log.txt");
      return new EnginePlayerDef(engineDef, limit);
    }


    /// <summary>
    /// Sets up the Ceres execution environment to expose a trained network 
    /// as a custom evaluator that can be referenced using the "CUSTOM1" identifier.
    /// 
    /// Because the set of reachable positions is in general not closed wrt the initial piece set (due to promotions),
    /// we actually install a "dynamic" evaluator which combines two evaluators:
    ///   - the trained network to be used for positions with the specified pieces (for which net was trained)
    ///   - as a fallback for other positions, a separate LC0 network which can play all positions.
    /// </summary>
    /// <param name="netFN"></param>
    /// <param name="posGenerator"></param>
    /// <param name="lc0NetToUseForUncoveredPositions"></param>
    /// <param name="ceresDeviceSpec"></param>
    public static void InstallCustomEvaluator(ICeresNeuralNetDef netDef, in ConfigNetExecution execConfig, 
                                              string netFN, PositionGenerator posGenerator,
                                              string lc0NetToUseForUncoveredPositions, string ceresDeviceSpec)
    {
      // Create evaluator for the specified trained neural network and also a fallback LC0 network.
      NNEvaluator evaluatorCeres = GetNNEvaluator(netDef, in execConfig,  netFN, true);
      if (lc0NetToUseForUncoveredPositions != null)
      {
        NNEvaluator evaluatorLC0 = NNEvaluator.FromSpecification(lc0NetToUseForUncoveredPositions, ceresDeviceSpec);

        // Create a "dynamic" evaluator which will use the Ceres network for positions
        // with the specified pieces (for which net was trained), otherwise the LC0 network which can play all positions.
        // This is necessary because the set of reachable positions is not closed over the initial piece set (due to promotions).
        evaluatorCeres = new NNEvaluatorDynamicByPos([evaluatorCeres, evaluatorLC0], (pos, _) => posGenerator.PositionMatches(in pos) ? 0 : 1);
      }

      // Install a handler to map the "CUSTOM1" evaluator to our evaluator for this network.
      NNEvaluatorFactory.Custom1Factory = (_, _, _) => evaluatorCeres;
    }



    static void WriteFloatRedIfNegative(float f, bool ok)
    {
      if (float.IsNaN(f))
      {
        Console.Write("".PadLeft(3));
      }
      else if (!ok)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, f.ToString("0.##").PadLeft(6), false);
      }
      else
      {
        Console.Write(f.ToString("0.##").PadLeft(6));
      }
    }


  }
}
