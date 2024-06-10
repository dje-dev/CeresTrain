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
using Ceres.Chess.LC0NetInference;
using Chess.Ceres.NNEvaluators;
using System.Runtime.InteropServices;
using CeresTrain.Networks.Transformer;
using CeresTrain.TrainCommands;
using CeresTrain.UserSettings;
using Ceres.Chess.UserSettings;
using System.Numerics.Tensors;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.CompilerServices;

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
        description+= $"_LE{upperBoundCount}";
        SKIP_COUNT = CeresTrainGenerateTPGDefault.SKIP_COUNT_IF_FILTERED_POSITIONS;
      }
      else if (pieces != null)
      {
        posPredicate = (Position pos) => pieces.PositionMatches(pos);
        description+= $"_{piecesString}";
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
    /// <param name="engineType"></param>
    /// <param name="netDef"></param>
    /// <param name="execConfig"></param>
    /// <param name="netFN"></param>
    /// <param name="useBestValueRepetitionHeuristic"></param>
    /// <returns></returns>
    public static NNEvaluator GetNNEvaluator(NNEvaluatorInferenceEngineType engineType,
                                             ICeresNeuralNetDef netDef, 
                                             int deviceID, in ConfigNetExecution execConfig,
                                             string netFN, bool useBestValueRepetitionHeuristic,
                                             object options)
    {
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

      NNEvaluatorTorchsharp evaluator = new(engineType, netDef, execConfig with {DeviceIDs = [deviceID], 
       
                                            SaveNetwork1FileName = netFN }, execConfig.Device, execConfig.DataType,
                                            options:(NNEvaluatorTorchsharpOptions) options);
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
        NNEvaluatorTorchsharpOptions options = default; // TO DO: fill this in
        InstallCustomEvaluator(1, engineType, "cuda", 0, netDef, execConfig, netFN, posGenerator, opponentNetID, ceresDeviceSpec, execConfig.UseHistory, options);
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
      NNEvaluatorTorchsharpOptions options = default; // TO DO: fill this in

      InstallCustomEvaluator(1, NNEvaluatorInferenceEngineType.CSharpViaTorchscript, "cuda", 0, netDef, execConfig, netFN, posGenerator,
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
                                               int deviceID,
                                               bool useHistory,
                                               string net1ReplacementNumStepsNetToUse = null,
                                               string net2ReplacementNumStepsNetToUse = null,
                                               bool monitorStats = false,
                                               NNEvaluatorTorchsharpOptions options = default,
                                               string netFNOverride = null,
                                               string netFNOverride2 = null)
    {
#if NOT
      // Automatically strip off extension ".onnx" if found since this is appended back on later.
      if (engineType == NNEvaluatorInferenceEngineType.ONNXRuntime
        || engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16
        || engineType == NNEvaluatorInferenceEngineType.ONNXRuntimeTensorRT
        || engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16TensorRT)
      {
        if (netFNOverride != null && netFNOverride.ToLower().EndsWith(".onnx"))
        {
          netFNOverride = netFNOverride.Substring(0, netFNOverride.IndexOf(".onnx"));
        }
        if (netFNOverride2 != null && netFNOverride2.ToLower().EndsWith(".onnx"))
        {
          netFNOverride2 = netFNOverride2.Substring(0, netFNOverride2.IndexOf(".onnx"));
        }
      }
#endif
      string configsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "configs");
      string fullConfigPath = Path.Combine(configsDir, configID);
      TrainingResultSummary? resultsFile;
      ConfigTraining config = default;
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
        netFileNameBase = resultsFile.Value.NetFileName;
        netFileNameBase = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "nets", Path.GetFileName(netFileNameBase.Replace("\\", "/")));
      }

      string netFileName1 = netFileNameBase;
      string netFileName2 = null; // Default is not to use a second net.

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

      InstallCustomEvaluator(evaluatorIndex,
                             engineType,
                             deviceType, deviceID,
                             config.NetDefConfig,
                             config.ExecConfig with
                             {
                               MonitorActivationStats = monitorStats,
                               SaveNetwork1FileName = netFileName1,
                               SaveNetwork2FileName = netFileName2,
                             },
                             netFileName1, null, null, null, useHistory, options);
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
                                              ICeresNeuralNetDef netDef, 
                                              ConfigNetExecution execConfig, 
                                              string netFN, 
                                              PositionGenerator posGenerator,
                                              string lc0NetToUseForUncoveredPositions,
                                              string lc0DeviceToUseForUncoveredPositions,
                                              bool useHistory,
                                              NNEvaluatorTorchsharpOptions evaluatorOptions)
    {
      ArgumentNullException.ThrowIfNullOrEmpty(netFN);

      execConfig = execConfig with { UseHistory = useHistory, 
                                     ActivationMonitorDumpSkipCount = evaluatorOptions.MonitorActivations ? 1 : 0,
                                     DeviceType = deviceType,
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
        EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true; // ** TODO: remove/rework
        NNEvaluatorEngineONNX.ConverterToFlatFromTPG = (o, f1)
          => TPGConvertersToFlat.ConvertToFlatTPGFromTPG(o, evaluatorOptions.QNegativeBlunders, evaluatorOptions.QPositiveBlunders, f1);
        NNEvaluatorEngineONNX.ConverterToFlat = (o, history, squares, legalMoveIndices) 
          => TPGConvertersToFlat.ConvertToFlatTPG(o, evaluatorOptions.QNegativeBlunders, evaluatorOptions.QPositiveBlunders, history, squares, legalMoveIndices);

        NNEvaluatorPrecision PRECISION = engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16 
                                      || engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16TensorRT
                                      ? NNEvaluatorPrecision.FP16 : NNEvaluatorPrecision.FP32;
        bool USE_TRT = engineType == NNEvaluatorInferenceEngineType.ONNXRuntimeTensorRT
                    || engineType == NNEvaluatorInferenceEngineType.ONNXRuntime16TensorRT;
        const bool HAS_UNCERTAINTY = false; // someday conditionally enable this
        const bool ENABLE_PROFILING = false;
        bool USE_STATE = evaluatorOptions.UsePriorState;
        bool HAS_ACTION = evaluatorOptions.UseAction;

        string onnxFN;
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

        //CeresTrainingRunAnalyzer.DumpAndBenchmarkONNXNetInfo(onnxFN);

        getEvaluatorFunc = (string netID, int gpuID, object options) =>
        {
          return new NNEvaluatorEngineONNX(netID,
                                           onnxFN, null, NNDeviceType.GPU, gpuID, USE_TRT,
                                           ONNXRuntimeExecutor.NetTypeEnum.TPG, NNEvaluatorTorchsharp.MAX_BATCH_SIZE,
                                           PRECISION, true, true, HAS_UNCERTAINTY, HAS_ACTION, "policy", "value", "mlh", "unc", true,
                                           false, ENABLE_PROFILING, false, useHistory, evaluatorOptions,
                                           true, evaluatorOptions.ValueHead1Temperature, evaluatorOptions.ValueHead2Temperature, evaluatorOptions.FractionValueHead2,
                                           USE_STATE);
        };
      }
      else
      {
        getEvaluatorFunc = (string netID, int gpuID, object options) =>
        {
          string netFNToUse = netID == null ? netFN : netID;
          ConfigNetExecution execConfigToUse = execConfig with { UseHistory = useHistory };
          NNEvaluator evaluator = GetNNEvaluator(engineType, netDef, gpuID, execConfig, netFNToUse, useBestValueRepetitionHeuristic, evaluatorOptions);
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



  /// <summary>
  /// Static helper methods to convert TPGRecord to flat square format.
  /// </summary>
  public static class TPGConvertersToFlat
  {
    /// <summary>
    /// Converts a batch of TPGRecord[] into TPG flat square values.
    /// </summary>
    /// <param name="records"></param>
    /// <param name="flatValues"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public static int ConvertToFlatTPGFromTPG(object records, float qNegativeBlunders, float qPositiveBlunders, byte[] flatValues)
    {
      // TODO: Requiring the converter to take a materialized array could be inefficient, can we use Memory instead?
      TPGRecord[] tpgRecords = records as TPGRecord[];
      if (tpgRecords == null)
      {
        throw new NotImplementedException("Expected input to be TPGRecord[]");
      }

      byte[] squareBytesAll = new byte[tpgRecords.Length * Marshal.SizeOf<TPGSquareRecord>() * 64];

      for (int i = 0; i < tpgRecords.Length; i++)
      {
        int offsetSquares = i * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD;

        // Extract as bytes.
        tpgRecords[i].CopySquares(squareBytesAll, offsetSquares);
      }

      // N.B. Scaling (by 100) already done in ONXNRuntimeExecutor (but probably doesn't belong there?).
      // TODO: This could be made more efficient, fold into the above loop.
      for (int i = 0; i < squareBytesAll.Length; i++)
      {
        flatValues[i] = squareBytesAll[i];
      }

      return squareBytesAll.Length;
    }


    /// <summary>
    /// Copies sourceBytes into targetFloats, also dividing by divisor.
    /// </summary>
    /// <param name="sourceBytes"></param>
    /// <param name="targetHalves"></param>
    static unsafe void CopyAndDivide(byte[] sourceBytes, Half[] targetHalves, float divisor)
    {
      const int CHUNK_SIZE = 1024 * 128;

      if (sourceBytes.Length < CHUNK_SIZE * 2)
      {
        CopyAndDivideSIMD(sourceBytes, targetHalves, divisor);
      }
      else
      { 
        Parallel.For(0, sourceBytes.Length / CHUNK_SIZE + 1, (chunkIndex) =>
        {
          int startIndex = chunkIndex * CHUNK_SIZE;
          int numThisBlock = Math.Min(CHUNK_SIZE, sourceBytes.Length - startIndex);

          if (numThisBlock > 0)
          {
            Span<byte> sourceBytesThisBlock = sourceBytes.AsSpan().Slice(startIndex, numThisBlock);
            Span<Half> targetHalvesThisBlock = targetHalves.AsSpan().Slice(startIndex, numThisBlock);
            CopyAndDivideSIMD(sourceBytesThisBlock, targetHalvesThisBlock, divisor);
          }
        });
      }

    }


    /// <summary>
    /// Copies sourceBytes into targetFloats, also dividing by divisor.
    /// </summary>
    /// <param name="sourceBytes"></param>
    /// <param name="targetHalfs"></param>
    static unsafe void CopyAndDivideSIMD(Span<byte> sourceBytes, Span<Half> targetHalfs, float divisor)
    {
      int vectorSize = Vector256<byte>.Count;
      int i = 0;

      if (Avx2.IsSupported)
      {
        Vector256<float> divisorVec = Vector256.Create(divisor);
        ushort* ptrTargetHalfs = (ushort *)Unsafe.AsPointer(ref targetHalfs[0]); // pinned just below

        fixed (byte* squareBytesAllPtr = sourceBytes)
        fixed (Half* flatValuesPrimaryPtr = targetHalfs)
        {
          // Process in chunks of 32 bytes
          for (i = 0; i <= sourceBytes.Length - vectorSize; i += vectorSize)
          {
            // Load 32 bytes from the byte array
            Vector256<byte> byteVec = Avx.LoadVector256(&squareBytesAllPtr[i]);

            // Convert the 32 bytes to 32 floats
            Vector256<short> ushortLow = Avx2.ConvertToVector256Int16(byteVec.GetLower());
            Vector256<float> floatLowLow = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(ushortLow.GetLower()));
            Vector256<float> floatLowHigh = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(ushortLow.GetUpper()));
            Vector256<uint> r1 = SingleToHalfAsWidenedUInt32_Vector256(floatLowLow, divisorVec);
            Vector256<uint> r2 = SingleToHalfAsWidenedUInt32_Vector256(floatLowHigh, divisorVec);
            Vector256<ushort> source2 = Vector256.Narrow(r1, r2);
            Avx.Store(&ptrTargetHalfs[i], source2);

            Vector256<short> ushortHigh = Avx2.ConvertToVector256Int16(byteVec.GetUpper());
            Vector256<float> floatHighLow = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(ushortHigh.GetLower()));
            Vector256<float> floatHighHigh = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(ushortHigh.GetUpper()));
            Vector256<uint> r1H = SingleToHalfAsWidenedUInt32_Vector256(floatHighLow, divisorVec);
            Vector256<uint> r2H = SingleToHalfAsWidenedUInt32_Vector256(floatHighHigh, divisorVec);
            Vector256<ushort> source2H = Vector256.Narrow(r1H, r2H);

            // Store the results
            Avx.Store(&ptrTargetHalfs[i + 16], source2H);
          }
        }
      }
#if NOT
      else if (AdvSimd.IsSupported)
      {
        throw new NotImplementedException("this code is not yet tested");

        Vector128<float> divisorVec = Vector128.Create(divisor);
        Vector128<float> reciprocalDivisorVec = AdvSimd.ReciprocalEstimate(divisorVec);
        fixed (byte* sourceBytesPtr = sourceBytes)
        fixed (Half* targetFloatsPtr = targetFloats)
        {
          for (i = 0; i <= sourceBytes.Length - vectorSize; i += vectorSize)
          {
            // Load 16 bytes from the byte array
            Vector128<byte> byteVec = AdvSimd.LoadVector128(&sourceBytesPtr[i]);

            // Split the vector into two 64-bit parts
            Vector64<byte> byteVecLower = byteVec.GetLower();
            Vector64<byte> byteVecUpper = byteVec.GetUpper();

            // Zero extend the lower and upper parts
            Vector128<ushort> ushortVecLower = AdvSimd.ZeroExtendWideningLower(byteVecLower);
            Vector128<ushort> ushortVecUpper = AdvSimd.ZeroExtendWideningLower(byteVecUpper);

            // Zero extend the widened vectors to 32-bit integers
            Vector128<uint> uintVecLower1 = AdvSimd.ZeroExtendWideningLower(ushortVecLower.GetLower());
            Vector128<uint> uintVecLower2 = AdvSimd.ZeroExtendWideningUpper(ushortVecLower);
            Vector128<uint> uintVecUpper1 = AdvSimd.ZeroExtendWideningLower(ushortVecUpper.GetLower());
            Vector128<uint> uintVecUpper2 = AdvSimd.ZeroExtendWideningUpper(ushortVecUpper);

            // Convert the 32-bit integers to single-precision floats
            Vector128<float> floatVecLower1 = AdvSimd.ConvertToSingle(uintVecLower1);
            Vector128<float> floatVecLower2 = AdvSimd.ConvertToSingle(uintVecLower2);
            Vector128<float> floatVecUpper1 = AdvSimd.ConvertToSingle(uintVecUpper1);
            Vector128<float> floatVecUpper2 = AdvSimd.ConvertToSingle(uintVecUpper2);

            // Divide by the divisor using reciprocal approximation
            Vector128<float> resultVecLower1 = AdvSimd.Multiply(floatVecLower1, reciprocalDivisorVec);
            Vector128<float> resultVecLower2 = AdvSimd.Multiply(floatVecLower2, reciprocalDivisorVec);
            Vector128<float> resultVecUpper1 = AdvSimd.Multiply(floatVecUpper1, reciprocalDivisorVec);
            Vector128<float> resultVecUpper2 = AdvSimd.Multiply(floatVecUpper2, reciprocalDivisorVec);

            // Store the results
            AdvSimd.Store(&targetFloatsPtr[i], resultVecLower1);
            AdvSimd.Store(&targetFloatsPtr[i + 4], resultVecLower2);
            AdvSimd.Store(&targetFloatsPtr[i + 8], resultVecUpper1);
            AdvSimd.Store(&targetFloatsPtr[i + 12], resultVecUpper2);
          }
        }
      }
#endif
      // Process remaining elements (15x slower than vectorized).
      for (; i < sourceBytes.Length; i++)
      {
        targetHalfs[i] = (Half)(sourceBytes[i] / divisor);
      }

    }

    /// <summary>
    /// Converts a float (after a division by a constant) to a half-precision float using vectorized instructions.
    /// 
    /// Code heavily based on .NET runtime (System.Numerics.Tensors.TensorPrimitives)
    /// </summary>
    /// <param name="value"></param>
    /// <param name="divisorVec"></param>
    /// <returns></returns>
    static Vector256<uint> SingleToHalfAsWidenedUInt32_Vector256(Vector256<float> value, Vector256<float> divisorVec)
    {
      value = Avx.Divide(value, divisorVec);
      Vector256<uint> vector8 = value.AsUInt32();
      Vector256<uint> vector9 = Vector256.ShiftRightLogical(vector8 & Vector256.Create(2147483648u), 16);
      Vector256<uint> vector10 = Vector256.Equals(value, value).AsUInt32();
      value = Vector256.Abs(value);
      value = Vector256.Min(Vector256.Create(65520f), value);
      Vector256<uint> vector11 = Vector256.Max(value, Vector256.Create(947912704u).AsSingle()).AsUInt32();
      vector11 &= Vector256.Create(2139095040u);
      vector11 += Vector256.Create(109051904u);
      value += vector11.AsSingle();
      vector8 = value.AsUInt32();
      Vector256<uint> vector12 = ~vector10 & Vector256.Create(31744u);
      vector8 -= Vector256.Create(1056964608u);
      Vector256<uint> vector13 = Vector256.ShiftRightLogical(vector8, 13);
      vector8 &= vector10;
      vector8 += vector13;
      vector8 &= ~vector12;
      Vector256<uint> vector14 = vector12 | vector9;
      return vector8 | vector14;
    }

    /// <summary>
    /// Converts a IEncodedPositionBatchFlat of encoded positions into TPG flat square values.
    /// </summary>
    /// <param name="batch"></param>
    /// <param name="includeHistory"></param>
    /// <param name="squareValues"></param>
    /// <param name="flatValuesSecondary"></param>
    /// <exception cref="NotImplementedException"></exception>
    public static void ConvertToFlatTPG(IEncodedPositionBatchFlat batch,
                                        float qNegativeBlunders, float qPositiveBlunders, 
                                        bool includeHistory, Half[] squareValues, short[] legalMoveIndices)
    {
      if (TPGRecord.EMIT_PLY_SINCE_LAST_MOVE_PER_SQUARE)
      {
        throw new NotImplementedException();
      }

      const bool EMIT_PLY_SINCE = false;
      TPGRecord tpgRecord = default;
      EncodedPositionBatchFlat ebf = batch as EncodedPositionBatchFlat;

      // TODO: Consider possibly restoring the commented out code below 
      //       to efficiently decode the two top positions into TPGRecord
      //       instead of having to setting EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true
      //       and incurring all that overhead.
      //       If do this, the regression/equivalency test can to be to compare
      //       this version computed here against the new more efficient code.
      // TODO: someday handle since ply, does that need to be passed in from the search engine?

      byte[] squareBytesAll;
      byte[] moveBytesAll;
      // TODO: consider pushing the CopyAndDivide below into this next method
      TPGRecordConverter.ConvertPositionsToRawSquareBytes(batch, includeHistory, batch.Moves, EMIT_PLY_SINCE, 
                                                          qNegativeBlunders, qPositiveBlunders,
                                                          out _, out squareBytesAll, legalMoveIndices);

      // TODO: push this division onto the GPU
      CopyAndDivide(squareBytesAll, squareValues, TPGSquareRecord.SQUARE_BYTES_DIVISOR);

#if OLD_TPG_COMBO_DIRECT_CONVER
      static bool HAVE_WARNED = false;

      int offsetAttentionInput = 0;
      int offsetBoardInput = 0;
      for (int i = 0; i < batch.NumPos; i++)
      {
        Position pos = batch.Positions[i].ToPosition;

        TPGRecordCombo tpgRecordCombo = default;

        // NOTE: history (prior move to square) not passed here
        //        var lastMoveInfo = batch[i].PositionWithBoards.LastMoveInfoFromSideToMovePerspective();
        //        Console.WriteLine(lastMoveInfo.pieceType + " " + lastMoveInfo.fromSquare + " " + lastMoveInfo.toSquare + " " + (lastMoveInfo.wasCastle ? " ************ " : ""));
        //        int? targetSquareFromPriorMoveFromOurPerspective = lastMoveInfo.pieceType == PieceType.None ? null : lastMoveInfo.toSquare.SquareIndexStartA1;
        int? targetSquareFromPriorMoveFromOurPerspective = null;
        if (!HAVE_WARNED)
        {
          HAVE_WARNED = true;
          Console.WriteLine("WARNING: ConvertToFlatTPG does not yet set history (via targetSquareFromPriorMoveFromOurPerspective), someday pass in IEncodedPositionBatchFlat somehow");
        }


        // Get first board
        int startOffset = i * 112;
        Span<BitVector64> bvOurs0   = MemoryMarshal.Cast<ulong, BitVector64>(batch.PosPlaneBitmaps.Slice(startOffset, 6));
        Span<BitVector64> bvTheirs0 = MemoryMarshal.Cast<ulong, BitVector64>(batch.PosPlaneBitmaps.Slice(startOffset + 6, 6));
        EncodedPositionBoard eb0 = new EncodedPositionBoard(bvOurs0, bvTheirs0, false).Mirrored;

        // Get second board
        startOffset = i * 112 + 13;
        Span<BitVector64> bvOurs1 = MemoryMarshal.Cast<ulong, BitVector64>(batch.PosPlaneBitmaps.Slice(startOffset, 6));
        Span<BitVector64> bvTheirs1 = MemoryMarshal.Cast<ulong, BitVector64>(batch.PosPlaneBitmaps.Slice(startOffset + 6, 6));
        EncodedPositionBoard eb1 = new EncodedPositionBoard(bvOurs1, bvTheirs1, false).Mirrored;

        (PieceType pieceType, Square fromSquare, Square toSquare, bool wasCastle) = EncodedPositionWithHistory.LastMoveInfoFromSideToMovePerspective(in eb0, in eb1);
//Console.WriteLine("decode_LAST_MOVE " + pieceType + " " + fromSquare + " " + toSquare + " " + wasCastle + " " + pos.FEN + " " + eb1.GetFEN(pos.IsWhite) + " " + eb1.GetFEN(!pos.IsWhite));



        if (pieceType != PieceType.None)
        {
          targetSquareFromPriorMoveFromOurPerspective = pos.IsWhite ? toSquare.SquareIndexStartA1
                                                                    : toSquare.Reversed.SquareIndexStartA1;
        }

        EncodedPositionBatchFlat batchFlat = batch as EncodedPositionBatchFlat; 
        TPGRecordConverter.ConvertToTPGCombo(in pos, targetSquareFromPriorMoveFromOurPerspective, false, default, ref tpgRecordCombo);

        // TODO: Consider if we could simplify and avoid code duplication, use this method
        //   TPGRecordConverter.ConvertToTPGCombo(in EncodedTrainingPosition trainingPos, ref TPGRecordCombo tpgRecordCombo)
        // like
        //   TPGRecordConverter.ConvertToTPGCombo(in batchFlat.PositionsBuffer[i].BoardsHistory, ref tpgRecordCombo)

        float[] rawDataSquaresAndMoves = tpgRecordCombo.SquareAndMoveRawValues;
        for (int j = 0; j < rawDataSquaresAndMoves.Length; j++)
        {
          flatValuesPrimary[offsetAttentionInput++] = rawDataSquaresAndMoves[j];
        }

        if (TPGRecordCombo.NUM_RAW_BOARD_BYTES_TOTAL > 0)
        {
          TPGWriter.ExtractRawBoard(batchFlat.PositionsBuffer[i].Mirrored, ref tpgRecordCombo);
          float[] rawBoardValues = tpgRecordCombo.BoardRawValues;
          for (int j = 0; j < rawBoardValues.Length; j++)
          {
            flatValuesSecondary[offsetBoardInput++] = rawBoardValues[j];
          }
        }

      }
#endif
    }


#if BUGGY
    /// <summary>
    /// Converts a batch of encoded positions into TPG combo format values (floats)
    /// ready to then be sent into neural network.
    /// </summary>
    /// <param name="batch"></param>
    /// <param name="flatValues"></param>
    static void ConvertToFlatTPG(IEncodedPositionBatchFlat batch, float[] flatValues)
    {
      Parallel.For(0, batch.NumPos, i =>
      {
        int offset = i * TPGRecordCombo.BYTES_PER_MOVE_AND_SQUARE_RECORD;

        Position pos = batch.Positions[i].ToPosition;

        TPGRecordCombo tpgRecordCombo = default;
        TPGRecordConverter.ConvertToTPGCombo(in pos, false, default, ref tpgRecordCombo);

        float[] rawData = tpgRecordCombo.RawBoardInputs;
        for (int j = 0; j < rawData.Length; j++)
        {
          flatValues[offset + j] = rawData[j];
        }
      });

    }
#endif

  }
}


#if NOT
          //          if (bestMoveNNEncoded.Move.IndexNeuralNet == 97 || bestMoveNNEncoded.Move.IndexNeuralNet == 103)
          if (pos.MiscInfo.EnPassantFileIndex != PositionMiscInfo.EnPassantFileIndexEnum.FileNone)
          {
            Console.WriteLine("................................ " + bestMoveTraining + " " + bestMoveNN);
          }

          //          Console.WriteLine("\r\n" + pos.FEN + "  " + bestMoveTraining + " " + bestMoveNN.Move);
          //          Console.WriteLine(evalResult.Policy);

          int promotionCount = 0;
          int epCount = 0;
          for (int ix = 0; ix < 64; ix++)
          {
            if (rawPosBuffer[i].SquaresAndMoves[ix].MoveRecord.PromotionBytes[0] > 0
             || rawPosBuffer[i].SquaresAndMoves[ix].MoveRecord.PromotionBytes[1] > 0)
              promotionCount++;
            if (rawPosBuffer[i].SquaresAndMoves[ix].SquareRecord.IsEnPassant > 0)
            {
              epCount++;
            }
          }

          if (promotionCount > 0)
          {
            Console.WriteLine(i + " found promotion " + pos.FEN);
          }
          if (epCount > 0)
          {
            Console.WriteLine(i + " found en passant " + pos.FEN);
          }
}
#endif




