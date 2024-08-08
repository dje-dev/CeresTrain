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
using System.Text.Json;
using System.Runtime.InteropServices;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.Positions;
using Ceres.Chess.Textual;
using Ceres.Features.Tournaments;

using CeresTrain.PositionGenerators;
using CeresTrain.Trainer;
using CeresTrain.Examples;
using CeresTrain.UserSettings;
using CeresTrain.NNEvaluators;
using Ceres.Chess.NNEvaluators.Ceres;

#endregion

namespace CeresTrain.TrainCommands
{
  /// <summary>
  /// Static class containing methods for processing all top-level CeresTrain commands.
  /// </summary>
  public static class CeresTrainCommands
  {
    /// <summary>
    /// Executes an "init" command to create a new ConfigTraining serialize to disk as JSON files.
    /// </summary>
    /// <param name="config"></param>
    /// <param name="configName"></param>
    public static void ProcessInitCommand(in ConfigTraining config, string configName)
    {
      ArgumentNullException.ThrowIfNull(config, nameof(config));
      CeresTrainInitialization.InitializeCeresTrainEnvironment();

      string configsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "configs");
      Directory.CreateDirectory(configsDir);
      TrainingHelpers.WriteConfigJSON(in config, configsDir, configName);
      Console.WriteLine();
      Console.WriteLine($"New configuration with base name {configName} written to current {configsDir}.");
    }


    /// <summary>
    /// Executes "info" command to dump full description of specified config (from JSON files) to console.
    /// </summary>
    /// <param name="config"></param>
    /// <param name="configsDir"></param>
    public static void ProcessInfoCommand(string config, string configsDir)
    {
      CeresTrainInitialization.InitializeCeresTrainEnvironment();

      if (CeresTrainCommandUtils.CheckConfigFound(config, configsDir))
      {
        ConfigSerializationJSON.DumpJSONToConsole(configsDir, config);
      }
    }


    /// <summary>
    /// Executes the train command with specified parameters.
    /// </summary>
    /// <param name="configID"></param>
    /// <param name="piecesStr"></param>
    /// <param name="numPos"></param>
    /// <param name="hostConfig"></param>
    /// <param name="devices"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public static TrainingResultSummary ProcessTrainCommand(string configID, string piecesStr,
                                                            long? numPos,
                                                            string hostName, string? tpgDir, int[] devices,
                                                            TrainingStatusTable trainingStatusTable)
    {
      CeresTrainInitialization.InitializeCeresTrainEnvironment();

      CeresTrainHostConfig hostConfig = default;
      if (hostName != null)
      {
        if (!CeresTrainHostConfig.RegisteredConfigs.TryGetValue(hostName, out hostConfig))
        {
          throw new Exception($"Host {hostName} not known (case sensitive), register in TrainingHostConfig.RegisteredConfigs");
        };
      }

      return ProcessTrainCommand(configID, piecesStr, numPos, tpgDir, hostConfig, devices, trainingStatusTable);
    }


    public static TrainingResultSummary ProcessTrainCommand(string configID, string piecesStr, long? numPos, string? tpgDir,
                                                            CeresTrainHostConfig hostConfig, int[] devices,
                                                            TrainingStatusTable trainingStatusTable)
    {
      string configsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "configs");
      TrainingResultSummary result = default;
      if (CeresTrainCommandUtils.CheckConfigFound(configID, configsDir))
      {

        string configBaseName = Path.Combine(configsDir, configID);

        if (hostConfig.HostName == null) // local
        {
          // Run locally.
          ConfigTraining adjustedConfig = TrainingHelpers.AdjustAndLoadConfig(configBaseName, piecesStr, devices);
           
          adjustedConfig = adjustedConfig with { OptConfig  = adjustedConfig.OptConfig with { NumTrainingPositions = numPos ?? adjustedConfig.OptConfig.NumTrainingPositions }, 
                                                 DataConfig = adjustedConfig.DataConfig with { TrainingFilesDirectory = tpgDir ?? adjustedConfig.DataConfig.TrainingFilesDirectory} };
          result = CeresTrainLauncher.RunLocalCSharp(configID, piecesStr, in adjustedConfig, trainingStatusTable);
        }
        else if (hostConfig.HostName.ToUpper() == "WSL")
        {
          if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
          {
            ConsoleUtils.WriteLineColored(ConsoleColor.Red, "WSL option only supported on Windows");
            throw new Exception();
          }

          int[] overrideDevice = [0];
          ConfigTraining adjustedConfig = TrainingHelpers.AdjustAndLoadConfig(configBaseName, piecesStr,
                                                                              devices, hostConfig.OverridePyTorchCompileMode);
          result = CeresTrainLauncher.RunRemoteWSL(hostConfig.CeresTrainPyDir, configID, hostConfig.PathToOutputFromHost, in adjustedConfig, trainingStatusTable);
        }
        else
        {
          ConfigTraining configRemote = TrainingHelpers.AdjustAndLoadConfig(configBaseName, piecesStr,
                                                                            devices, hostConfig.OverridePyTorchCompileMode);

          string pathToConfigFromHost = hostConfig.PathToOutputFromHost + "/configs/" + configID;

          result = CeresTrainLauncher.RunRemoteSSH(hostConfig.HostName, hostConfig.UserName, hostConfig.CeresTrainPyDir,
                                                   configID, pathToConfigFromHost, in configRemote, hostConfig.PathToOutputFromHost, 
                                                   hostConfig.DockerLaunchCommand, trainingStatusTable);
        }

        string resultsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "results");
        string resultsFileName = result.WriteJSON(resultsDir, configID);
        Console.WriteLine();
        Console.WriteLine("Summary of results written to file: " + resultsFileName);
        Console.WriteLine("  " + File.ReadAllText(resultsFileName));
      }

      return result;
    }

    /// <summary>
    /// Executes "UCI" command to launch UCI command loop using specified config and pieces.
    /// </summary>
    /// <param name="configID"></param>
    /// <param name="piecesStr"></param>
    /// <param name="configsDir"></param>
    /// <param name="netSpecForUncoveredPositions">optional specification of a network to be used for use in uncovered positions wrt. piecesStr</param>
    public static void ProcessUCICommand(string configID, string piecesStr, string configsDir, string netSpecForUncoveredPositions)
    {
      CeresTrainInitialization.InitializeCeresTrainEnvironment();

      if (CeresTrainCommandUtils.CheckConfigFound(configID, configsDir))
      {
        TrainingResultSummary? resultsFile = CeresTrainCommandUtils.ReadResultsForConfig(configID);
        if (resultsFile == null)
        {
          return;
        }

        string fullConfigPath = Path.Combine(configsDir, configID);
        ConfigTraining config = TrainingHelpers.AdjustAndLoadConfig(fullConfigPath, piecesStr);
        CeresNetEvaluation.RunUCILoop(config.NetDefConfig, config.ExecConfig, resultsFile.Value.NetFileName, netSpecForUncoveredPositions, "GPU:0", null);
      }
    }


    static (string id, NNEvaluator evaluator) cachedEvaluator = default;

    /// <summary>
    /// Tests specified LC0 network accuracy on positions with specified set of pieces.
    /// </summary>
    /// <param name="piecesStr"></param>
    /// <param name="netID"></param>
    /// <param name="numPos"></param>
    /// <param name="verbose"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public static (float valueAccuracy, float policyAccuracy) ProcessEvalLC0Command(string piecesStr, 
                                                                                    string netID, 
                                                                                    long numPos, 
                                                                                    string epdOrPGNFileName = null,
                                                                                    SearchLimit searchLimit = default,
                                                                                    bool verbose = false)
    {
      CeresTrainInitialization.InitializeCeresTrainEnvironment();

      if (numPos > int.MaxValue)
      {
        throw new ArgumentException("numPos too large");
      }

      NNEvaluator evaluator = cachedEvaluator.id == netID ? cachedEvaluator.evaluator : NNEvaluator.FromSpecification(netID, "GPU:0");
      cachedEvaluator = (netID, evaluator);

      Console.WriteLine("Testing value/policy accuracy...");
      (float valueAccuracy, float policyAccuracy) = CeresNetEvaluation.TestNetValueAccuracy(evaluator, piecesStr, (int)numPos, epdOrPGNFileName, verbose);

      // Run a tournament against itself with tablebases enabled.      
      Console.WriteLine();
      TournamentResultStats tournResult = default;
      if (searchLimit != default)
      {
        tournResult = RunEvalOrTournament(null, piecesStr, numPos, epdOrPGNFileName, verbose, netID, true, null, true, searchLimit);
      }

      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, "Tested net  " + netID + " on " + piecesStr + " with " + numPos + " positions and tournament at " + searchLimit);
      Console.WriteLine($"  Value  accuracy  : {100f * valueAccuracy:F2}%");
      Console.WriteLine($"  Policy accuracy  : {100f * policyAccuracy:F2}%");
      if (searchLimit != default)
      {
        Console.WriteLine($"  vs. Tablebases   : {tournResult.ShortEloSummaryStr} Elo");
      }
      return (valueAccuracy, policyAccuracy);
    }


    /// <summary>
    /// Processes either the EVAL or TOURN commands, which compare
    /// either a value/policy accuracy comparison or run a value head tournament between 
    /// a Ceres net and a specified LC0 network.
    /// 
    /// TODO: someday break into 2 methods (one for eval, on for tournament).
    /// </summary>
    /// <param name="configID"></param>
    /// <param name="piecesStr"></param>
    /// <param name="numPos"></param>
    /// <param name="verbose"></param>
    /// <param name="compareLC0NetSpec"></param>
    /// <param name="opponentTablebasesEnabled"></param>
    /// <param name="configsDir"></param>
    /// <param name="runTournament"></param>
    /// <param name="searchLimit"></param>
    /// <exception cref="Exception"></exception>
    public static TournamentResultStats RunEvalOrTournament(string configID, string piecesStr, long numPos, string sourceEPDOrPGNFileName,
                                                            bool verbose, string compareLC0NetSpec, bool opponentTablebasesEnabled,
                                                            string configsDir, bool runTournament, SearchLimit searchLimit)
    {
      CeresTrainInitialization.InitializeCeresTrainEnvironment();

      if (configID == null || CeresTrainCommandUtils.CheckConfigFound(configID, configsDir))
      {
        if (numPos > int.MaxValue)
        {
          throw new Exception("numPos too large");
        }

        TrainingResultSummary? resultsFileInfo = null;
        ConfigTraining config = default;
        if (configID != null)
        {
          resultsFileInfo = CeresTrainCommandUtils.ReadResultsForConfig(configID);
          if (resultsFileInfo == null)
          {
            return default;
          }

          // N.B. The TorchSharp evaluator (first line) MUST be initialized
          //      before the LC0 evaluator (second line) to avoid a crash (CUDA initialization).
          // TODO: research and try to fix this.
          config = TrainingHelpers.AdjustAndLoadConfig(Path.Combine(configsDir, configID), piecesStr);

          Console.WriteLine();
          Console.WriteLine("LOADING: " + resultsFileInfo.Value.NetFileName 
            + " pos=" + resultsFileInfo.Value.NumTrainingPositions 
            + " loss=" + resultsFileInfo.Value.LossSummary.TotalLoss
            + " value accuracy=" + resultsFileInfo.Value.LossSummary.ValueAccuracy);
        }

        PositionGeneratorRandomFromPieces generator = new PositionGeneratorRandomFromPieces(piecesStr);

        string netFileName = configID == null ? null : resultsFileInfo.Value.NetFileName;
        ICeresNeuralNetDef netDefConfig = configID == null ? default : config.NetDefConfig;
        ConfigNetExecution execConfig = configID == null ? default : config.ExecConfig;

        if (runTournament)
        {
          if (compareLC0NetSpec == null)
          {
            if (opponentTablebasesEnabled)
            {
              // The actual neural network specified won't matter since we're running against tablebases.
              // Use the random evaluator to avoid loading a network.
              compareLC0NetSpec = "RANDOM_WIDE:";
            }
            else
            {
              throw new Exception("net-id option must be specified for tournament (to indicate network against which to run the tournament");
            }
          }        

          PositionGenerator posGeneratorForTournament =
            sourceEPDOrPGNFileName == null ? new PositionGeneratorFromIEnumerable(piecesStr, pos => generator.PositionMatches(in pos), generator.AsPositionEnumerable())
                                           : new PositionGeneratorFromIEnumerable(piecesStr+"_file", pos => generator.PositionMatches(in pos), PositionsWithHistory.FromEPDOrPGNFile(sourceEPDOrPGNFileName, (int)numPos, p => generator.PositionMatches(in p))); 

          TournamentResultStats tournResults = CeresNetEvaluation.RunTournament(NNEvaluatorInferenceEngineType.CSharpViaTorchscript, netDefConfig, execConfig, netFileName,
                                                                                compareLC0NetSpec, "GPU:0", posGeneratorForTournament, searchLimit, (int)numPos, verbose,
                                                                                opponentTablebasesEnabled);
          return tournResults;
        }
        else
        {
          if (configID == null)
          {
            throw new NotImplementedException();
          }
          string FN = sourceEPDOrPGNFileName;
          NNEvaluatorOptionsCeres options = default; // TODO: fill in
          NNEvaluator evaluator = CeresNetEvaluation.GetNNEvaluator(NNEvaluatorInferenceEngineType.CSharpViaTorchscript, netDefConfig, 0, execConfig, netFileName, true, options);
          NNEvaluator compareLC0Evaluator = compareLC0NetSpec == null ? null : NNEvaluator.FromSpecification(compareLC0NetSpec, "GPU:0");
          (float accuracyValue, float accuracyPolicy) = CeresNetEvaluation.TestAccuracyOnPositions(generator, FN, evaluator, compareLC0Evaluator, default, (int)numPos, verbose);
        }
      }

      return default;
    }


    /// <summary>
    /// Extracts a set of positions matching specified pieces from an EPD or PGN file into a new EPD or PGN file.
    /// </summary>
    /// <param name="epdOrPGNFileName"></param>
    /// <param name="pieces"></param>
    /// <param name="outFileName"></param>
    /// <param name="numPositions"></param>
    public static void ExtractToEPD(string epdOrPGNFileName, PieceList pieces, string outFileName, int numPositions)
    {
      Console.WriteLine($"Beginning extraction from {epdOrPGNFileName}...");
      PGNExtractor.ExtractFromPGN(epdOrPGNFileName, outFileName, pieces.ToPredicate, numPositions);
      Console.WriteLine($"Extracted {numPositions} positions to {outFileName}");
    }

  }
}
