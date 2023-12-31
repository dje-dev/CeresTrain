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
using System.Diagnostics;
using System.IO;

using Renci.SshNet;

using Ceres.Base.Misc;

using CeresTrain.Trainer;
using CeresTrain.UserSettings;
using CeresTrain.Utils;

#endregion

namespace CeresTrain.TrainCommands
{
  /// <summary>
  /// Static top-level coordinator methods for launching and monitoring training 
  /// on the local machine (direct in TorchSharp/C# or via PyTorch/Python in WSL)
  /// or a remote machine (in PyTorch/Python).
  /// </summary>
  public static class CeresTrainLauncher
  {
    /// <summary>
    /// Runs training configuration locally using TorchSharp (in C#).
    /// </summary>
    /// <param name="configID"></param>
    /// <param name="piecesStr"></param>
    /// <param name="config"></param>
    /// <returns></returns>
    public static TrainingResultSummary RunLocalCSharp(string configID, string piecesStr, in ConfigTraining config)
    {
      DateTime startTime = DateTime.Now;
      string hostName = Environment.MachineName;
      string configPath = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "configs", configID);

      (FileLogger logger, TrainingStatusTable consoleStatusTable) = CeresTrainCommandUtils.DoTrainingPrologue(hostName, null, configID, in config, configPath);

      string trainingDescription = $"Local train of config {configID} on {piecesStr} for {config.OptConfig.NumTrainingPositions} pos on device {string.Join(",", config.ExecConfig.DeviceIDs)}";
      CeresTrainCommandTrain boardTrain = new(config, trainingDescription);
      TrainingResultSummary trainResult = boardTrain.DoTrain(trainingDescription);
      trainResult = trainResult with { TrainingLogFileName = logger.LiveLogFileName };

      // Add INFO lines to the logger (which is otherwise used for local training)
      // so that supplemental information can be parsed out for use in the TrainingResultSummary.
      logger.AddLine($"INFO: TORCHSCRIPT_FILENAME {trainResult.NetFileName}");
      logger.AddLine($"INFO: NUM_PARAMETERS {boardTrain.NumParameters}");
      logger.AddLine(END_TRAINING_PHRASE + " SUCCESS");
      CeresTrainCommandUtils.DoTrainingEpilogue(Environment.MachineName, configID, startTime, logger, consoleStatusTable, trainResult);
      return trainResult;
    }



    /// <summary>
    /// Runs PyTorch training code locally from a Windows machine using WSL.
    /// </summary>
    /// <param name="ceresTrainPyDir"></param>
    /// <param name="configID"></param>
    /// <param name="hostPathToOutput"></param>
    /// <param name="config"></param>
    /// <returns></returns>
    public static TrainingResultSummary RunRemoteWSL(string ceresTrainPyDir, string configID, string hostPathToOutput, in ConfigTraining config)
    {
      string configFullPath = $"{hostPathToOutput}/configs/{configID}";

      (FileLogger logger, TrainingStatusTable consoleStatusTable) = CeresTrainCommandUtils.DoTrainingPrologue("wsl", ceresTrainPyDir, configID, in config, configFullPath);
      DateTime startTime = DateTime.Now;

      string netOutputPath = $"{hostPathToOutput}/nets";

      ProcessStartInfo startInfo = new()
      {
        FileName = "wsl",
        Arguments = $"bash -c \"cd {ceresTrainPyDir} && python3 train.py {configFullPath} {netOutputPath}\"",
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        CreateNoWindow = true
      };

      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, "Launching on WSL (localhost): " + startInfo.Arguments);

      using Process process = new Process { StartInfo = startInfo };
      process.Start();

      int numTrainLinesSeen = 0;

      Console.WriteLine();
      consoleStatusTable.RunTraining(() =>
      {
        while (!process.StandardOutput.EndOfStream)
        {
          string line = process.StandardOutput.ReadLine();
          logger.AddLine(line);

          CeresTrainCommandUtils.UpdateTableWithLine(consoleStatusTable, ref startTime, ref numTrainLinesSeen, line);

          if (line.Contains(END_TRAINING_PHRASE))
          {
            break; // Training complete.
          }

        }

        // Read any remaining lines from STDERR
        string lineErr;
        while ((lineErr = process.StandardError.Peek() == -1 ? null : process.StandardError.ReadLine()) != null)
        {
          logger.AddLine("STDERR: " + lineErr);
        }

      });

      process.WaitForExit();

      return CeresTrainCommandUtils.DoTrainingEpilogue("wsl", configID, startTime, logger, consoleStatusTable);
    }



    /// <summary>
    /// Runs PyTorch training code on a (potentially) remote Linux machine over SSH.
    /// </summary>
    /// <param name="hostName"></param>
    /// <param name="userName"></param>
    /// <param name="hostWorkingDir"></param>
    /// <param name="configID"></param>
    /// <param name="configPath"></param>
    /// <param name="config"></param>
    /// <param name="saveNetDirectory"></param>
    /// <returns></returns>
    public static TrainingResultSummary RunRemoteSSH(string hostName, string userName, string hostWorkingDir, string configID,
                                                     string configPath, in ConfigTraining config, string saveNetDirectory)
    {
      (FileLogger logger, TrainingStatusTable consoleStatusTable) = CeresTrainCommandUtils.DoTrainingPrologue(hostName, hostWorkingDir, configID, in config, configPath);

      DateTime startTime = DateTime.Now;

      consoleStatusTable.RunTraining(() =>
      {
        DoGoRemote(consoleStatusTable, logger, hostName, userName, hostWorkingDir, configPath, saveNetDirectory);
      });

      return CeresTrainCommandUtils.DoTrainingEpilogue(hostName, configID, startTime, logger, consoleStatusTable);
    }


    #region Internal helpers

    // End of training will be recognized by this string.
    const string END_TRAINING_PHRASE = "INFO: EXIT_STATUS";

    static void DoGoRemote(TrainingStatusTable table, FileLogger logger,
                          string hostName, string userName, string baseDir,
                          string configID, string saveNetDirectory)
    {
      using SSHClient sshClientx = new SSHClient(hostName, userName);
      sshClientx.CheckSSHClientConnected();

      SshClient sshClient = sshClientx.SSHExecClient;
      sshClient.Connect();
      Console.WriteLine();
      Console.WriteLine("Connected to " + hostName);

      int numTrainLinesSeen = 0;
      DateTime startTime = default;

      string command = $"cd {baseDir} && python3 train.py {configID} {saveNetDirectory}";

      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"Launching on {hostName} as {userName} via SSH: " + command);

      using (ShellStream shellStream = sshClient.CreateShellStream("CeresTrainPy_remote_" + Environment.MachineName, 80, 24, 800, 600, 1024))
      {
        // Writing command to the shell stream and executing it
        shellStream.WriteLine(command);

        // Final command to exit the shell.
        shellStream.WriteLine("exit");

        Console.WriteLine("Beginning initialization/execution of train.py...");
        Console.WriteLine();
        int numLinesRead = 0;
        while (true)
        {

          string line = null;
          try
          {
            line = shellStream.ReadLine(TimeSpan.FromSeconds(0.05f));
          }
          catch (Exception)
          {
            if (numLinesRead == 0)
            {
              ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"Immediate error/disconnected received from host {hostName} with command {command}");
              Environment.Exit(3);
            }
            else
            {
              ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"After {numLinesRead} lines received from host {hostName}, error/disconnected received. See log file at {logger.LiveLogFileName}");
              Environment.Exit(3);
            }
          }

          numLinesRead++;
          if (line == null)
          {
            continue; // Timeout or no more data
          }
          else
          {
            logger.AddLine(line);
            CeresTrainCommandUtils.UpdateTableWithLine(table, ref startTime, ref numTrainLinesSeen, line);

            if (line.StartsWith(END_TRAINING_PHRASE))
            {
              break; // Training complete.
            }
          }
        }
      }

    }

    #endregion

  }
}
