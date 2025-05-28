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
using Ceres.Chess.Positions;
using Ceres.Base.Benchmarking;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Zstandard.Net;
using Ceres.Base.DataType;
using Ceres.Chess.EncodedPositions;
using CeresTrain.CeresTrainDefaults;
using Ceres.Base.Math;
using Chess.Ceres.NNEvaluators;
using CeresTrain.Networks.Transformer;
using CeresTrain.TrainCommands;
using CeresTrain.UserSettings;
using Ceres.Chess.UserSettings;
using System.Numerics.Tensors;
using System.Runtime.Intrinsics.Arm;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.NNEvaluators.Ceres;
using Ceres.Chess.NNBackends.ONNXRuntime;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.GameEngines;
using Newtonsoft.Json.Serialization;
using Spectre.Console.Rendering;
using BenchmarkDotNet.Attributes;
using Ceres.Base.DataTypes;
using Ceres.MCTS.Evaluators;
using static TorchSharp.torch;
using Ceres.Chess.MoveGen;

#endregion 

namespace CeresTrain.Examples
{
  /// <summary>
  /// Set of static methods to facilitate running various tests of Ceres neural networks.
  /// </summary>
  public static class CeresNetEvaluation
  {
    /// <summary>
    /// Generates a set of TPG files from LC0 training games stored in TAR or ZST training data files.
    /// </summary>
    /// <param name="piecesString"></param>
    /// <param name="numPositions"></param>
    /// <param name="tarDirectory"></param>
    /// <param name="outputDirectory"></param>
    public static void GenerateTPGFilesFromLC0TrainingData(string piecesString, long numPositions, string tarDirectory, string outputDirectory)
    {
      if (!Directory.Exists(tarDirectory))
      {
        throw new Exception($"Specified TAR directory {tarDirectory} does not exist.");
      }

      if (!Directory.Exists(outputDirectory))
      {
        Directory.CreateDirectory(outputDirectory);
      }

      PieceList pieces = null;
      bool usePieceCountLimit = false;
      int upperBoundCount = 0;

      if (!string.IsNullOrEmpty(piecesString))
      {
        if (piecesString.StartsWith("<") && int.TryParse(piecesString.Substring(1), out upperBoundCount))
        {
          usePieceCountLimit = true;
        }
        else
        {
          pieces = new PieceList(piecesString);
        }
      }

      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"GenerateTPGFilesFromLC0TrainingData positions {numPositions} "
                                                     + $"of {piecesString} from {tarDirectory} threads to {outputDirectory}");

      string description = "CT";
      int SKIP_COUNT = CeresTrainGenerateTPGDefault.SKIP_COUNT_IF_ALL_POSITIONS;
      Predicate<Position> posPredicate = null;
      if (usePieceCountLimit)
      {
        posPredicate = (Position pos) => pos.PieceCount <= upperBoundCount;
        description += $"_LE{upperBoundCount}";
        SKIP_COUNT = CeresTrainGenerateTPGDefault.SKIP_COUNT_IF_FILTERED_POSITIONS;
      }
      else if (pieces != null)
      {
        posPredicate = (Position pos) => pieces.PositionMatches(pos);
        description += $"_{piecesString}";
        SKIP_COUNT = CeresTrainGenerateTPGDefault.SKIP_COUNT_IF_FILTERED_POSITIONS;
      }


      // Launch TPG generation.
      DateTime startTime = DateTime.Now;
      using (new TimingBlock("GenerateTPGFilesFromLC0TrainingData"))
      {
        const long BLOCK_SIZE = 8192 * 1640;
        numPositions = MathUtils.RoundedUp(numPositions, BLOCK_SIZE);
        CeresTrainGenerateTPGDefault.GenerateTPG(tarDirectory, outputDirectory, numPositions, description,
                                                   (EncodedTrainingPositionGame game, int positionIndex, in Position position)
                                                   => posPredicate(position), null, SKIP_COUNT);

      }

      DateTime endTime = DateTime.Now;
      int nps = (int)((float)(numPositions) / (endTime - startTime).TotalSeconds);
      Console.WriteLine();
      Console.WriteLine($"Done: {nps:N0} positions per second generation to TPG files.");
    }


    /// <summary>
    /// Generates a set of TPG files from random endgame positions having specified set of pieces.
    /// </summary>
    /// <param name="piecesString"></param>
    /// <param name="numPositions"></param>
    /// <param name="outputDirectory"></param>
    public static void GenerateTPGFilesFromTablebasePositions(string piecesString, 
                                                              long numPositions, 
                                                              string outputDirectory)
    {
      if (!Directory.Exists(outputDirectory))
      {
        Directory.CreateDirectory(outputDirectory);
      }

      // Construct a PieceList from the pieces string (verifies validity).
      //PieceList pieceList = new(piecesString); TODO: not yet updated for multi syntax like [KRPkrp,0.2]

      int NUM_THREADS = 4 + (int)MathF.Min(16, Environment.ProcessorCount / 2);
      const int BATCH_SIZE = 1024;
      long NUM_BATCHES_TO_WRITE = numPositions / (NUM_THREADS * BATCH_SIZE);
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"GenerateTPGFileFromTablebasePositions positions {numPositions} "
                                                     + $"of {piecesString} with {NUM_THREADS} threads to {outputDirectory}");

      // Launch and wait for parallel threads, each writing to separate file.
      DateTime startTime = DateTime.Now;
      const bool SUCCEED_IF_INCOMPLETE_DTZ_INFORMATION = true;
      using (new TimingBlock("GenerateTPGFilesFromTablebasePositions"))
      {
        List<Task> tasks = new();
        Enumerable.Range(0, NUM_THREADS).ToList().ForEach(i =>
        {
          tasks.Add(Task.Run(() => GenerateTPGFileFromTablebasePositions(SUCCEED_IF_INCOMPLETE_DTZ_INFORMATION, outputDirectory, piecesString, i, BATCH_SIZE, NUM_BATCHES_TO_WRITE)));
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
    /// <param name="filenameIndex"></param>
    /// <param name="batchSize"></param>
    /// <param name="numBatches"></param>
    static void GenerateTPGFileFromTablebasePositions(bool succeedIfIncompleteDTZInformation, 
                                                      string outputDirectory, 
                                                      string piecesString, 
                                                      int filenameIndex, 
                                                      int batchSize, 
                                                      long numBatches)
    {
      PositionGeneratorRandomFromPieces generator = 
        piecesString.Contains("[") ? PositionGeneratorRandomFromPieces.CreateFromMultiPiecesStr(piecesString) // e.g. "[KQPkqp,0.2],[KRPkrp,0.8]"
                                   : new PositionGeneratorRandomFromPieces(piecesString);
      TablebaseTPGBatchGenerator tpgGenerator = new(generator.ID, generator.GeneratePosition, succeedIfIncompleteDTZInformation, batchSize);

      string outFN = Path.Combine(outputDirectory, @$"{FileUtils.FileNameSanitized(generator.ID)}_{Random.Shared.Next()}_{filenameIndex}.dat.zst");

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


    /// <summary>
    /// Method that returns a neural network evaluator for a given trained network on disk. 
    /// </summary>
    /// <param name="engineType"></param>
    /// <param name="netDef"></param>
    /// <param name="execConfig"></param>
    /// <param name="netFN"></param>
    /// <param name="useBestValueRepetitionHeuristic"></param>
    /// <returns></returns>
    public static NNEvaluatorTorchsharp GetNNEvaluator(NNEvaluatorInferenceEngineType engineType,
                                                       ICeresNeuralNetDef netDef,
                                                       Device device, ScalarType dataType,
                                                       int deviceID, in ConfigNetExecution execConfig,
                                                       string netFN, bool useBestValueRepetitionHeuristic,
                                                       object options)
    {
      if (dataType == ScalarType.BFloat16)
      {
        throw new Exception("BFloat16 not recommended due to low number of mantissa bits, may create inaccuracies");
      }

      // If present, fixup the net filename to include the path to Ceres nets.
      if (netFN != null && !File.Exists(netFN))
      {
        string orgNetFN = netFN;
        netFN = Path.Combine(CeresUserSettingsManager.Settings.DirCeresNetworks, netFN);
        if (!File.Exists(netFN))
        {
          throw new Exception($"Ceres net file {orgNetFN} not found in DirCeresNetworks {CeresUserSettingsManager.Settings.DirCeresNetworks}");
        }
      }

      NNEvaluatorTorchsharp evaluator = new(engineType, execConfig with
        {
          DeviceIDs = [deviceID],
          SaveNetwork1FileName = netFN,
          DeviceType  = device.type.ToString(),
          DataType = dataType,            
      }, device, dataType,
        options: (NNEvaluatorOptionsCeres)options,
        netTransformerDef: engineType == NNEvaluatorInferenceEngineType.CSharpViaTorchscript ?(NetTransformerDef)netDef : default);

      evaluator.UseBestValueMoveUseRepetitionHeuristic = useBestValueRepetitionHeuristic;
      return evaluator;
    }


    /// <summary>
    /// Run a match between the specified trained network and the LC0 reference network,
    /// using either the pure value or pure policy head outputs.
    /// </summary>
    /// <param name="engineType"></param>
    /// <param name="netFN"></param>
    /// <param name="opponentNetID"></param>
    /// <param name="ceresDeviceSpec"></param>
    /// <param name="posGenerator"></param>
    /// <param name="searchLimit"></param>
    /// <param name="numGamePairs"></param>
    /// <param name="verbose"></param>
    public static TournamentResultStats RunTournament(NNEvaluatorInferenceEngineType engineType,
                                                      ICeresNeuralNetDef netDef, in ConfigNetExecution execConfig,
                                                      string netFN, string opponentNetID, string ceresDeviceSpec,
                                                      PositionGenerator posGenerator,
                                                      SearchLimit searchLimit,
                                                      int numGamePairs = 50, bool verbose = false,
                                                      bool opponentTablebasesEnabled = false)
    {
      EnginePlayerDef player1;
      EnginePlayerDef player2;

      if (netFN != null)
      {
        // Playing Ceres net versus LC0 net (with or without tablebases, as determined by opponentTablebasesEnabled).
        NNEvaluatorOptionsCeres options = default; // TO DO: fill this in
        InstallCustomEvaluator(1, engineType, "cuda", 0, TorchSharp.torch.ScalarType.BFloat16, netDef, execConfig, netFN, posGenerator, opponentNetID, ceresDeviceSpec, execConfig.UseHistory, options);
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
      const bool SF_ENABLE = false;
      const bool SF_WITH_TABLEBASE = false;
      const int SF_NUM_THREADS = 10;
      const float SF_SEARCH_TIME_SECS = 0.3f;
      GameEngineUCI engineSF = SF_ENABLE ? SFEvaluatorPool.CreateSFEngine(SF_NUM_THREADS, SF_WITH_TABLEBASE) : null;

      bool FILL_IN_HISTORY = !evaluatorPrimary.UseBestValueMoveUseRepetitionHeuristic; // Some LC0 nets seem to require; Ceres nets will override to ignore

      ISyzygyEvaluatorEngine tbEvaluator = ISyzygyEvaluatorEngine.DefaultEngine;

      int numValueCorrectPrimary = 0;
      int numPolicyCorrectPrimary = 0;
      int numValueCorrectCompare = 0;
      int numPolicyCorrectCompare = 0;
      int numValueCorrectSF = 0;
      int numPolicyCorrectSF = 0;

      int numMispredicitedDecisivePrimary = 0;
      int numMispredicitedDecisiveCompare = 0;
      int numMispredictedDecisiveSF = 0;

      int numMispredicitedDrawPrimary = 0;
      int numMispredicitedDrawCompare = 0;
      int numMispredicitedDrawSF = 0;

      IEnumerable<PositionWithHistory> positionSourceEnum = sourceEPDOrPGN == null
                                            ? generator.AsPositionWithHistoryEnumerable()
                                            : PositionsWithHistory.FromEPDOrPGNFile(sourceEPDOrPGN, numPos, null, p => generator.PositionMatches(in p)).AsEnumerable();
      IEnumerator<PositionWithHistory> posEnumerator = positionSourceEnum.AsEnumerable().GetEnumerator();

      PositionWithHistory[] positions = new PositionWithHistory[numPos];

      // TODO: This is a common pattern that could be factored out:
      //         -- build a batch of Positions (maybe from PGN)
      //         -- evaluate all (Oversized)
      //         -- extract as NNEvaluatorResult
      EncodedPositionBatchBuilder batchBuilder = new EncodedPositionBatchBuilder(numPos, NNEvaluator.InputTypes.All);
      for (int i = 0; i < numPos; i++)
      {
        if (!posEnumerator.MoveNext())
        {
          throw new Exception($"Insufficient positions matching {generator} found in file {sourceEPDOrPGN}");
        }

        // N.B. We truncate any history, since endgame training tool sourcing positions
        //      from tablebases will not have had history available during training.
        PositionWithHistory pos = positions[i] = new(posEnumerator.Current.FinalPosition);

        batchBuilder.Add(pos, FILL_IN_HISTORY);
      }


      EncodedPositionBatchFlat batch = batchBuilder.GetBatch();

      NNEvaluatorResult[] resultsCeres = new NNEvaluatorResult[numPos];
      NNEvaluatorResult[] resultsCompare = evaluatorCompare == null ? null : new NNEvaluatorResult[numPos];

      // ########################################
      if (false && SF_ENABLE)
      {
        IPositionEvaluationBatch bufferedBatch = evaluatorPrimary.EvaluateIntoBuffers(batch, false);
        PositionEvaluationBatch bufferedBatchDirect = bufferedBatch as PositionEvaluationBatch;

        for (int i = 0; i < numPos; i++)
        {
          //NNEvaluatorResult thisResult = batchEvaluated[i];
          float scoreQ = SFEvaluatorPool.StockfishSearch(positions[i].FinalPosition, new SearchLimit(SearchLimitType.SecondsPerMove, SF_SEARCH_TIME_SECS));
          //GameEngineSearchResult sfResult = engineSF.Search(positions[i], new SearchLimit(SearchLimitType.SecondsPerMove, 0.3f));
          Console.Write(bufferedBatchDirect.GetWin1P(i) + " " + bufferedBatchDirect.GetLoss1P(i) + " --> ");
          bufferedBatchDirect.SetWL(i, scoreQ);
          Console.WriteLine(scoreQ + " --> " + bufferedBatchDirect.GetWin1P(i) + " " + bufferedBatchDirect.GetLoss1P(i));
        }
        // ########################################
      }

      evaluatorPrimary.EvaluateOversizedBatch(batch, (int index, NNEvaluatorResult result) => { resultsCeres[index] = result; });
      evaluatorCompare?.EvaluateOversizedBatch(batch, (int index, NNEvaluatorResult result) => { resultsCompare[index] = result; });

      int[,] countActualPredictedPrimary = new int[3, 3];
      int[,] countActualPredictedCompare = new int[3, 3];
      int[,] countActualPredictedSF = new int[3, 3];
      int[] gameResultFrequency = new int[3];
      for (int i = 0; i < numPos; i++)
      {
        // N.B. We truncate any history, since endgame training tool sourcing positions
        //      from tablebases will not have had history available during training.
        PositionWithHistory pos = positions[i];

        NNEvaluatorResult evalResult = resultsCeres[i];
        NNEvaluatorResult resultCompare = evaluatorCompare == null ? default : resultsCompare[i];

        //GameEngineSearchResult sfResult = null;
        float scoreQ;
        if (SF_ENABLE)
        {
          //scoreQ = SFEvaluatorPool.StockfishSearch(positions[i].FinalPosition, new SearchLimit(SearchLimitType.SecondsPerMove, SF_SEARCH_TIME_SECS));
          GameEngineSearchResult sfResult = engineSF.Search(positions[i], new SearchLimit(SearchLimitType.SecondsPerMove, SF_SEARCH_TIME_SECS));
          scoreQ = sfResult.ScoreQ;
          //scoreQ = SFEvaluatorPool.StockfishSearch(pos.FinalPosition, new SearchLimit(SearchLimitType.SecondsPerMove, SF_SEARCH_TIME_SECS));
        }

        int gameResultTablebase = tbEvaluator.ProbeWDLAsV(pos.FinalPosition);
        gameResultFrequency[gameResultTablebase + 1]++;
        bool netValueCorrect = gameResultTablebase == evalResult.MostProbableGameResult;
        bool compareOK = resultCompare.MostProbableGameResult == gameResultTablebase;
        int sfMostProbableGameResult = SF_ENABLE ? (scoreQ > 0.5f ? 1 : (scoreQ < -0.5 ? -1 : 0)) : 0;
        bool sfCorrect = gameResultTablebase == sfMostProbableGameResult;

        countActualPredictedPrimary[gameResultTablebase + 1, evalResult.MostProbableGameResult + 1]++;
        countActualPredictedCompare[gameResultTablebase + 1, resultCompare.MostProbableGameResult + 1]++;
        countActualPredictedSF[gameResultTablebase + 1, sfMostProbableGameResult + 1]++;

        bool netTopMoveInBestCategory = tbEvaluator.MoveIsInOptimalCategoryForPosition(pos.FinalPosition, evalResult.Policy.TopMove(pos.FinalPosition), true);
        bool netTopMoveInBestCategoryCompare = evaluatorCompare == null ? false : tbEvaluator.MoveIsInOptimalCategoryForPosition(pos.FinalPosition, resultCompare.Policy.TopMove(pos.FinalPosition), true);

        if (verbose)// && (!netValueCorrect || !compareOK))
        {
          string gameResultString = gameResultTablebase == 1 ? "Win " : (gameResultTablebase == -1 ? "Loss" : "Draw");
          Console.Write($"{gameResultString}  Primary=");
          WriteFloatRedIfNegative(evalResult.V, netValueCorrect);

          Console.Write("  Comp=");
          WriteFloatRedIfNegative(evaluatorCompare == null ? float.NaN : resultCompare.V, compareOK);

          if (SF_ENABLE)
          {
            Console.Write("  SF=");
            WriteFloatRedIfNegative(SF_ENABLE == null ? float.NaN : scoreQ, sfCorrect);
          }
          Console.WriteLine($"  {evalResult.Policy,-115}   {pos.FinalPosition.FEN}");
        }

        numValueCorrectPrimary += netValueCorrect ? 1 : 0;
        numValueCorrectCompare += compareOK ? 1 : 0;
        numValueCorrectSF += sfCorrect ? 1 : 0;
        numPolicyCorrectPrimary += netTopMoveInBestCategory ? 1 : 0;
        numPolicyCorrectCompare += netTopMoveInBestCategoryCompare ? 1 : 0;
      }

      Console.WriteLine();
      Console.WriteLine();
      Console.WriteLine("TEST RESULTS " + sourceEPDOrPGN);
      Console.WriteLine();
      Console.WriteLine("Test net: Actual vs predicted");


      WriteTable("Test net", countActualPredictedCompare, gameResultFrequency);
      WriteTable("Compare", countActualPredictedCompare, gameResultFrequency);
      if (SF_ENABLE)
      {
        WriteTable("Stockfish", countActualPredictedSF, gameResultFrequency);
      }

      Console.WriteLine();
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

      if (SF_ENABLE)
      {
        Console.WriteLine($"  Accuracy value (SF)   : {(100.0f * ((float)numValueCorrectSF / numPos)):F2}% using {SF_SEARCH_TIME_SECS} secs with {SF_NUM_THREADS} threads");
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


    private static void WriteTable(string playerID, int[,] countActualPredictedCompare, int[] gameResultFrequency)
    {
      Console.WriteLine();
      Console.WriteLine($"{playerID}: Actual vs predicted");
      Console.WriteLine("                          Loss    Draw     Win");
      for (int ix = 0; ix < 3; ix++)
      {
        string actualString = ix == 0 ? "Loss " : (ix == 1 ? "Draw " : "Win  ");
        Console.Write($"Actual: {gameResultFrequency[ix],6:N0}   " + actualString);
        for (int jx = 0; jx < 3; jx++)
        {
          Console.Write($"{countActualPredictedCompare[ix, jx],8:N0}");
        }
        Console.WriteLine();
      }
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
                                            : PositionsWithHistory.FromEPDOrPGNFile(epdOrPGNFileName, numPositionsToTest, null, p => generator.PositionMatches(in p)).Select(s => s.FinalPosition).AsEnumerable();
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
          MGMove topMove = nnResults[i].Policy.TopMove(in batchPositions[i]);
          bool policyCorrect = !topMove.IsNull && tbEvaluator.MoveIsInOptimalCategoryForPosition(in batchPositions[i], topMove, true);
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
                            + $"  {batchPositions[i].FEN,-35}   {nnResults[i].V,8:F2} {nnResults[i].Policy}");
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
      NNEvaluatorOptionsCeres options = default; // TO DO: fill this in

      InstallCustomEvaluator(1, NNEvaluatorInferenceEngineType.CSharpViaTorchscript, "cuda", 0, TorchSharp.torch.ScalarType.BFloat16, netDef, execConfig, netFN, posGenerator,
                             lc0NetToUseForUncoveredPositions, ceresDeviceSpec, execConfig.UseHistory, options);
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
    /// Installs Ceres network as a custom evaluator (CUSTOM1 or CUSTOM2).
    /// </summary>
    /// <param name="evaluatorIndex"></param>
    /// <param name="configID"></param>
    public static void InstallCustomEvaluatorEx(int evaluatorIndex, string configID,
                                               NNEvaluatorInferenceEngineType engineType,
                                               string deviceType,
                                               TorchSharp.torch.ScalarType dataType,
                                               int deviceID,
                                               bool useHistory,
                                               string net1ReplacementNumStepsNetToUse = null,
                                               string net2ReplacementNumStepsNetToUse = null,
                                               bool monitorStats = false,
                                               NNEvaluatorOptionsCeres options = default,
                                               string netFNOverride = null,
                                               string netFNOverride2 = null)
    {
      throw new NotImplementedException();
#if NOT

      string netFileName1 = null;
      string netFileName2 = null;

      ConfigTraining config = default;
      if (configID != null)
      {
        string configsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "configs");
        string fullConfigPath = Path.Combine(configsDir, configID);
        TrainingResultSummary? resultsFile;
        if (!CeresTrainCommandUtils.CheckConfigFound(configID, configsDir))
        {
          throw new Exception($"Config not found: {configID} at {fullConfigPath}");
        }

        resultsFile = CeresTrainCommandUtils.ReadResultsForConfig(configID);
        if (resultsFile == null && netFNOverride == null)
        {
          throw new Exception("Unable to load results for " + configID);
        }

        string netFileNameBase;
        if (netFNOverride != null)
        {
          netFileNameBase = netFNOverride;
        }
        else
        {
          // Determine network full path, remapping saved path to directory under main output directory.
          netFileNameBase = resultsFile.Value.TorchscriptFileName;
          netFileNameBase = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "nets", Path.GetFileName(netFileNameBase.Replace("\\", "/")));
        }

        netFileName1 = netFileNameBase;
        netFileName2 = null; // Default is not to use a second net.

        // Potentially use a different saved net (at a specified number of steps).
        if (net1ReplacementNumStepsNetToUse != null)
        {
          netFileName1 = netFileName1.Replace("_final", $"_{net1ReplacementNumStepsNetToUse}");
        }

        // Potentially use a different saved net (at a specified number of steps).
        if (net2ReplacementNumStepsNetToUse != null)
        {
          netFileName2 = netFileNameBase.Replace("_final", $"_{net2ReplacementNumStepsNetToUse}");
        }

        if (netFNOverride2 != null)
        {
          netFileName2 = netFNOverride2;
        }


        config = TrainingHelpers.AdjustAndLoadConfig(fullConfigPath, "KPkp", null); // TODO: don't require pieces
        if (config.ExecConfig.SaveNetwork2FileName != null)
        {
          throw new NotImplementedException("SaveNetwork2FileName support not completed (how to reconcile netFileName above and this value?)");
        }
      }

      InstallCustomEvaluator(evaluatorIndex,
                             engineType,
                             deviceType, deviceID, dataType,
                             config.NetDefConfig,
                             config.ExecConfig with
                             {
                               MonitorActivationStats = monitorStats,
                               SaveNetwork1FileName = netFileName1,
                               SaveNetwork2FileName = netFileName2,
                             },
                             netFileName1, null, null, null, useHistory, options);
#endif
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
    /// <param name="engineType"></param>
    /// <param name="netFN"></param>
    /// <param name="posGenerator"></param>
    /// <param name="lc0NetToUseForUncoveredPositions"></param>
    /// <param name="lc0DeviceToUseForUncoveredPositions"></param>
    public static void InstallCustomEvaluator(int customEvaluatorIndex,
                                              NNEvaluatorInferenceEngineType engineType,
                                              string deviceType,
                                              int deviceIndex,
                                              TorchSharp.torch.ScalarType dataType,
                                              ICeresNeuralNetDef netDef,
                                              ConfigNetExecution execConfig,
                                              string netFN,
                                              PositionGenerator posGenerator,
                                              string lc0NetToUseForUncoveredPositions,
                                              string lc0DeviceToUseForUncoveredPositions,
                                              bool useHistory,
                                              NNEvaluatorOptionsCeres evaluatorOptions)
    {
      throw new NotImplementedException();
#if NOT
      execConfig = execConfig with
      {
        UseHistory = useHistory,
        ActivationMonitorDumpSkipCount = evaluatorOptions.MonitorActivations ? 1 : 0,
        DeviceType = deviceType,
        DataType = dataType,
        DeviceIDs = [deviceIndex]
      };
      bool useBestValueRepetitionHeuristic = !useHistory;
      // Create evaluator for the specified trained neural network and also a fallback LC0 network.
      //      NNEvaluator evaluatorCeres = GetNNEvaluator(netDef, in execConfig, netFN, useBestValueRepetitionHeuristic);

      NNEvaluator evaluatorCeres;

      Func<string, int, object, NNEvaluator> getEvaluatorFunc = null;


      if (engineType == NNEvaluatorInferenceEngineType.ONNXRuntime
        || engineType == NNEvaluatorInferenceEngineType.ONNXRuntimeTensorRT
        || engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16
        || engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16TensorRT
        )
      {
        NNEvaluatorPrecision PRECISION = engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16
                                      || engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16TensorRT
                                      ? NNEvaluatorPrecision.FP16 : NNEvaluatorPrecision.FP32;
        bool USE_TRT = engineType == NNEvaluatorInferenceEngineType.ONNXRuntimeTensorRT
                    || engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16TensorRT;
        const bool HAS_UNCERTAINTY_V = true;
        const bool HAS_UNCERTAINTY_P = true;
        const bool ENABLE_PROFILING = false;
        bool USE_STATE = evaluatorOptions.UsePriorState;
        bool HAS_ACTION = evaluatorOptions.UseAction;
        string onnxFN = null;

        if (netFN != null)
        {
          if (netFN.EndsWith(".ts"))
          {
            ConsoleUtils.WriteLineColored(ConsoleColor.Red, ".ts file provided but evaluator was configured as ONNX " + netFN);
          }

          if (netFN.ToUpper().EndsWith(".ONNX"))
          {
            // Take filename as is.
            onnxFN = netFN;
          }
          else
          {
            onnxFN = netFN + ".onnx";
            if (engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16 || engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16TensorRT)
            {
              onnxFN = onnxFN + "_fp16.onnx";
            }
          }

          if (!File.Exists(onnxFN))
          {
            if (netFN != null && !File.Exists(netFN))
            {
              string orgNetFN = onnxFN;
              onnxFN = Path.Combine(CeresUserSettingsManager.Settings.DirCeresNetworks, netFN);
              if (!File.Exists(onnxFN))
              {
                throw new Exception($"Ceres ONNX net file {orgNetFN} not found in DirCeresNetworks {CeresUserSettingsManager.Settings.DirCeresNetworks}");
              }
            }
          }
        }

        if (customEvaluatorIndex == 1)
        {
          NNEvaluatorFactory.Custom1Options = evaluatorOptions;
        }
        else if (customEvaluatorIndex == 2)
        {
          NNEvaluatorFactory.Custom2Options = evaluatorOptions;
        }
        else
        {
          throw new NotImplementedException();
        }

        //CeresTrainingRunAnalyzer.DumpAndBenchmarkONNXNetInfo(onnxFN);

        getEvaluatorFunc = (string netID, int gpuID, object options) =>
        {
          string useONNXFN = (netID != null && netID.ToUpper().EndsWith(".ONNX"))
                              ? Path.Combine(CeresUserSettingsManager.Settings.DirCeresNetworks, netID)
                              : onnxFN;
          NNEvaluatorOptionsCeres captureOptions = (NNEvaluatorOptionsCeres)(customEvaluatorIndex == 1 ? NNEvaluatorFactory.Custom1Options
                                                                                                                 : NNEvaluatorFactory.Custom2Options);
          NNEvaluatorONNX onnxEngine = new (netID, useONNXFN, null, NNDeviceType.GPU, gpuID, USE_TRT,
                                                  ONNXNetExecutor.NetTypeEnum.TPG, NNEvaluatorTorchsharp.MAX_BATCH_SIZE,
                                                   PRECISION, true, true, HAS_UNCERTAINTY_V, HAS_UNCERTAINTY_P, HAS_ACTION, "policy", "value", "mlh", "unc", true,
                                                   ENABLE_PROFILING, false, useHistory, captureOptions,
                                                  true, USE_STATE);

          EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true; // ** TODO: remove/rework
          onnxEngine.ConverterToFlatFromTPG = (options, o, f1)
            => TPGConvertersToFlat.ConvertToFlatTPGFromTPG(options, o, f1);
          onnxEngine.ConverterToFlat = (options, o, history, squares, legalMoveIndices)
            => TPGConvertersToFlat.ConvertToFlatTPG(options, o, history, squares, legalMoveIndices);

          return onnxEngine;
        };
      }
      else
      {
        getEvaluatorFunc = (string netID, int gpuID, object options) =>
        {
          string netFNToUse = netID == null ? netFN : netID;
          ConfigNetExecution execConfigToUse = execConfig with { UseHistory = useHistory, DataType = TorchSharp.torch.ScalarType.BFloat16 };
          NNEvaluator evaluator = GetNNEvaluator(engineType, netDef, gpuID, execConfigToUse, netFNToUse, useBestValueRepetitionHeuristic, evaluatorOptions);
          evaluator.Description = "CERES [" + evaluatorOptions.ShortStr + "] " + netFNToUse;
          return evaluator;
        };
      }

      Func<NNEvaluator, int, NNEvaluator> possiblyWrapEvaluatorFunc = (NNEvaluator innerEvaluator, int gpuID) =>
      {
        if (lc0NetToUseForUncoveredPositions != null)
        {
          NNEvaluator evaluatorLC0 = NNEvaluator.FromSpecification(lc0NetToUseForUncoveredPositions, lc0DeviceToUseForUncoveredPositions);

          // Create a "dynamic" evaluator which will use the Ceres network for positions
          // with the specified pieces (for which net was trained), otherwise the LC0 network which can play all positions.
          // This is necessary because the set of reachable positions is not closed over the initial piece set (due to promotions).
          return new NNEvaluatorDynamicByPos([innerEvaluator, evaluatorLC0], (pos, _) => posGenerator.PositionMatches(in pos) ? 0 : 1);
        }
        else
        {
          return innerEvaluator;
        }
      };

      // Install a handler to map the "CUSTOM<id>" evaluator to our evaluator for this network.
      if (customEvaluatorIndex == 1)
      {
        NNEvaluatorFactory.Custom1Factory = (string netID, int gpuID, NNEvaluator referenceEvaluator, object options)
          => possiblyWrapEvaluatorFunc(getEvaluatorFunc(netID, gpuID, options), gpuID);
      }
      else if (customEvaluatorIndex == 2)
      {
        NNEvaluatorFactory.Custom2Factory = (string netID, int gpuID, NNEvaluator referenceEvaluator, object options)
          => possiblyWrapEvaluatorFunc(getEvaluatorFunc(netID, gpuID, options), gpuID);
      }
      else
      {
        throw new Exception($"Invalid custom evaluator index {customEvaluatorIndex}");
      }

      string baseInfo = $"Installed CUSTOM{customEvaluatorIndex} evaluator {engineType} {netFN} history {execConfig.UseHistory}";
      if (posGenerator == null)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Blue, baseInfo);
      }
      else
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Blue, baseInfo + $", outside {posGenerator} uses {lc0NetToUseForUncoveredPositions}");
      }
    }


#endif
    }
  }
}
