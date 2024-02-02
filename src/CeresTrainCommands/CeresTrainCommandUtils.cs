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
using System.Collections;
using System.IO;
using System.Linq;
using System.Text.Json;

using Ceres.Base.Misc;

using CeresTrain.Trainer;
using CeresTrain.UserSettings;

#endregion

namespace CeresTrain.TrainCommands
{
  /// <summary>
  /// Miscellaneous static utility methods for CeresTrain commands.
  /// </summary>
  public static class CeresTrainCommandUtils
  {
    /// <summary>
    /// Creates and initializes a logger for the training process.
    /// </summary>
    /// <param name="configID"></param>
    /// <param name="configPath"></param>
    /// <returns></returns>
    internal static FileLogger CreateLogger(string configID, string configPath)
    {
      string logDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "logs");
      Directory.CreateDirectory(logDir);

      string fn = Environment.MachineName + "_" + configID + "_" + DateTime.Now.Ticks % 10_000 + ".txt";
      FileLogger logger = new FileLogger(logDir, fn, verbose: false, fileHeaderString: $"CeresTrainPy output for {configPath} on {Environment.MachineName} via WSL");
      Console.WriteLine("Writing log to " + logger.LiveLogFileName);
      Console.WriteLine();

      return logger;
    }


    /// <summary>
    /// Executes common preparatory step for a training session (e.g. file and console logging).
    /// </summary>
    /// <param name="trainingHost"></param>
    /// <param name="ceresTrainPyDir"></param>
    /// <param name="configID"></param>
    /// <param name="config"></param>
    /// <param name="configPath"></param>
    /// <returns></returns>
    internal static (FileLogger, TrainingStatusTable) DoTrainingPrologue(string trainingHost, string ceresTrainPyDir,
                                                                         string configID, in ConfigTraining config, string configPath,
                                                                         TrainingStatusTable trainingStatusTable)
    {
      ConfigSerializationJSON.DumpJSONToConsole(configID);
      Console.WriteLine();
      Console.WriteLine(trainingHost + " " + configID + " " + ceresTrainPyDir);

      config.Validate(ceresTrainPyDir == null);

      FileLogger logger = CreateLogger(configID, configPath);

      if (trainingStatusTable == null)
      {
        string infoStr = config.TrainingDescriptionStr(trainingHost, ceresTrainPyDir);
        trainingStatusTable = new TrainingStatusTable(configPath, infoStr, config.OptConfig.NumTrainingPositions, false);
      }

      lock (trainingStatusTable)
      {
        Console.WriteLine();
        ConsoleUtils.WriteLineColored(ConsoleColor.Cyan, $"Ceres Training on {trainingHost} of {config.NetDefConfig} net ");
        ConsoleUtils.WriteLineColored(ConsoleColor.Cyan, $"Config: {configPath} for {config.OptConfig.NumTrainingPositions:N0} positions"
                                                      + (ceresTrainPyDir == null ? "" : $"(in directory {ceresTrainPyDir})"));
      }

      return (logger, trainingStatusTable);
    }


    /// <summary>
    /// Executes common finalization steps for a training session (e.g. writing results JSON file).
    /// </summary>
    /// <param name="trainingHost"></param>
    /// <param name="startTime"></param>
    /// <param name="logger"></param>
    /// <param name="consoleStatusTable"></param>
    /// <param name="trainResult">optional result if already extracted</param>
    /// <returns></returns>
    internal static TrainingResultSummary DoTrainingEpilogue(string trainingHost, string configID, DateTime startTime, FileLogger logger,
                                                             TrainingStatusTable consoleStatusTable, TrainingResultSummary? trainResult = null)
    {
      return ExtractSummary(trainingHost, configID, logger, consoleStatusTable, startTime);
    }


    /// <summary>
    /// Constructs a TrainingResultSummary with information about the training session results.
    /// </summary>
    /// <param name="trainingHost"></param>
    /// <param name="configID"></param>
    /// <param name="logger"></param>
    /// <param name="consoleStatusTable"></param>
    /// <param name="startTime"></param>
    /// <param name="trainResult">optional result if already extracted</param>
    /// <returns></returns>
    static TrainingResultSummary ExtractSummary(string trainingHost, string configID, FileLogger logger, 
                                                TrainingStatusTable consoleStatusTable,
                                                DateTime startTime, TrainingResultSummary? trainResult = null)
    {
      logger.LoadInfoDictionary();

      string exitStatus = logger.GetInfoStr("EXIT_STATUS");
      if (exitStatus != "SUCCESS")
      {
        Console.WriteLine();

        Console.WriteLine("Training failed with exit status: " + (exitStatus ?? "(abended, unknown)"));
        logger.DumpErrorLines();
        Console.WriteLine("See log file for full details at  " + logger.LiveLogFileName);
        Console.WriteLine("Aborting.");
        Environment.Exit(1);
      }

      if (trainResult != null)
      {
        return trainResult.Value;
      }

      long numParameters = logger.GetInfoLong("NUM_PARAMETERS");
      string tsFileName = logger.GetInfoStr("TORCHSCRIPT_FILENAME");
      string netsDir = CeresTrainUserSettingsManager.Settings.OutputNetsDir;
      string savedTSFilename = tsFileName == null ? null : Path.Combine(netsDir, tsFileName);

      TrainingLossSummary lossSummary = default;
      TrainingStatusTable.TrainingStatusRecord[] records = consoleStatusTable.TrainingStatusRecords.Where(s => s.configID == configID).ToArray();
      if (records .Length > 0)
      {
        lossSummary = new()
        {
          PolicyAccuracy = records[^1].PolicyAcc,
          ValueAccuracy = records[^1].ValueAcc,
          TotalLoss = records[^1].TotalLoss,
          ValueLoss = records[^1].ValueLoss,
          PolicyLoss = records[^1].PolicyLoss,
          Value2Loss = records[^1].Value2Loss,
          MLHLoss = records[^1].MLHLoss,
          UNCLoss = records[^1].UNCLoss,
          QDeviationLowerLoss = records[^1].QDeviationLowerLoss,
          QDeviationUpperLoss = records[^1].QDeviationUpperLoss,
        };
      }

      TrainingStatusTable.TrainingStatusRecord last = records.Length > 0 ? records[^1] : default;
      TrainingResultSummary result = new TrainingResultSummary(Environment.MachineName, trainingHost, configID, DateTime.Now, exitStatus,
                                                               numParameters, DateTime.Now - startTime, last.NumPositions,
                                                               lossSummary, logger.LiveLogFileName, savedTSFilename, null);
      return result;
    }


    /// <summary>
    /// Method called each time a line is read from the training process output,
    /// used to potentially update the TrainingStatusTable.
    /// </summary>
    /// <param name="table"></param>
    /// <param name="startTime"></param>
    /// <param name="numTrainLinesSeen"></param>
    /// <param name="line"></param>
    internal static void UpdateTableWithLine(TrainingStatusTable table, string configID, ref DateTime startTime, ref int numTrainLinesSeen, string line)
    {
      if (line.StartsWith("TRAIN:"))
      {
        CeresTrainProgressLoggingLine trainingData = new CeresTrainProgressLoggingLine(line);
        if (trainingData.NumPos <= 4096)
        {
          return; // Skip early positions to avoid capturing initialization time.
        }

        if (numTrainLinesSeen++ == 0)
        {
          startTime = DateTime.Now;
        }

        float elapsedSeconds = (float)(DateTime.Now - startTime).TotalSeconds;
        table.UpdateInfo(DateTime.Now, configID, elapsedSeconds, trainingData.NumPos, trainingData.TotalLoss,
                         trainingData.LastValueLoss, 0.01f * trainingData.LastValueAcc,
                         trainingData.LastPolicyLoss, 0.01f * trainingData.LastPolicyAcc,
                         trainingData.LastMLHLoss, trainingData.LastUNCLoss, 
                         trainingData.LastValue2Loss, 
                         trainingData.LastQDeviationLowerLoss, trainingData.LastQDeviationUpperLoss,
                         trainingData.LastLR);
      }
    }


    /// <summary>
    /// Reads the most recent TrainingResultSummary corresponding to the specified configuration.
    /// </summary>
    /// <param name="config"></param>
    /// <returns></returns>
    public static TrainingResultSummary? ReadResultsForConfig(string config)
    {
      string resultsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "results");
      string resultsFileName = Path.Combine(resultsDir, config + "_results.json");
      if (!File.Exists(resultsFileName))
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"Results file {resultsFileName} not found. "
                                                       + "Network should be trained first yielding network and associated results file");
        return null;
      }

      TrainingResultSummary resultsFile = JsonSerializer.Deserialize<TrainingResultSummary>(File.ReadAllText(resultsFileName));

      return resultsFile;
    }


    internal const string NO_CONFIG_ERR_STR = "Configuration must be specified (--config=<config_name>).";

    /// <summary>
    /// Verifies that the specified configuration exists in the specified directory.
    /// </summary>
    /// <param name="config"></param>
    /// <param name="configsDir"></param>
    /// <returns></returns>
    public static bool CheckConfigFound(string config, string configsDir)
    {
      if (config == null)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, NO_CONFIG_ERR_STR);
        return false;
      }
      else if (!File.Exists(Path.Combine(configsDir, config + "_ceres_opt.json")))
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"Config {config} not found in directory {configsDir}");
        return false;
      }
      else
      {
        return true;
      }
    }

  }
}
