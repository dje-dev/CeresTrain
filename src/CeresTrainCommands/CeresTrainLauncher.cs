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
    public static TrainingResultSummary RunLocalCSharp(string configID, string piecesStr, in ConfigTraining config, TrainingStatusTable statusTable)
    {
      DateTime startTime = DateTime.Now;
      string hostName = Environment.MachineName;
      string configPath = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "configs", configID);

      (FileLogger logger, TrainingStatusTable consoleStatusTable) = CeresTrainCommandUtils.DoTrainingPrologue(hostName, null, configID, in config, configPath, statusTable);

      string trainingDescription = $"Local train of config {configID} on {piecesStr} for {config.OptConfig.NumTrainingPositions} pos on device {string.Join(",", config.ExecConfig.DeviceIDs)}";
      CeresTrainCommandTrain boardTrain = new(config, trainingDescription, consoleStatusTable);
      TrainingResultSummary trainResult = boardTrain.DoTrain(trainingDescription);
      trainResult = trainResult with { TrainingLogFileName = logger.LiveLogFileName };

      // Add INFO lines to the logger (which is otherwise used for local training)
      // so that supplemental information can be parsed out for use in the TrainingResultSummary.
      logger.AddLine($"INFO: TORCHSCRIPT_FILENAME {trainResult.TorchscriptFileName}");
      logger.AddLine($"INFO: NUM_PARAMETERS {boardTrain.NumParameters}");
      logger.AddLine(END_TRAINING_PHRASE + " SUCCESS");
      CeresTrainCommandUtils.DoTrainingEpilogue(Environment.MachineName, configID, startTime, logger, consoleStatusTable, trainResult);
      return trainResult;
    }


    static readonly object consoleOutLockObject = new();

    /// <summary>
    /// Runs PyTorch training code locally from a Windows machine using WSL.
    /// </summary>
    /// <param name="ceresTrainPyDir"></param>
    /// <param name="configID"></param>
    /// <param name="hostPathToOutput"></param>
    /// <param name="config"></param>
    /// <returns></returns>
    public static TrainingResultSummary RunPyTorchLocal(string hostName, string ceresTrainPyDir, string configID, string hostPathToOutput,
                                                       in ConfigTraining config, TrainingStatusTable trainingStatusTable)
    {
      bool isWSL = hostName.ToUpper() == "WSL";
      string configFullPath = $"{hostPathToOutput}/configs/{configID}";

      (FileLogger logger, TrainingStatusTable consoleStatusTable) = CeresTrainCommandUtils.DoTrainingPrologue(configID, ceresTrainPyDir, configID, in config, configFullPath, trainingStatusTable);
      DateTime startTime = DateTime.Now;

      ProcessStartInfo startInfo = new()
      {
        FileName = isWSL ? "wsl" : "bash",
        Arguments = $"bash -c \"cd {ceresTrainPyDir} && python3 train.py {configFullPath} {hostPathToOutput}\"",
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        CreateNoWindow = isWSL,
      };

      lock (consoleOutLockObject)
      {
        Console.WriteLine();
        ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"Launching on {hostName} (localhost): " + startInfo.Arguments);
      }

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

          CeresTrainCommandUtils.UpdateTableWithLine(consoleStatusTable, configID, configID, ref startTime, ref numTrainLinesSeen, line);

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

      return CeresTrainCommandUtils.DoTrainingEpilogue(hostName, configID, startTime, logger, consoleStatusTable);
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
    public static TrainingResultSummary RunPyTorchRemote(string hostName, string userName, string hostWorkingDir, string configID,
                                                         string configPath, in ConfigTraining config, string outputsDirectory,
                                                         string dockerLaunchCommand,
                                                         TrainingStatusTable trainingStatusTable)
    {
      (FileLogger logger, TrainingStatusTable consoleStatusTable) = CeresTrainCommandUtils.DoTrainingPrologue(hostName, hostWorkingDir, configID, in config, configPath, trainingStatusTable);

      DateTime startTime = DateTime.Now;

      if (config.ExecConfig.RunInDocker)
      {
        if (dockerLaunchCommand == null)
        {
          throw new Exception("Docker requested but host configuration is missing required launch command.");
        }
      }
      else
      {
        dockerLaunchCommand = null;
      }

      consoleStatusTable.RunTraining(() =>
      {
        DoGoRemote(consoleStatusTable, logger, hostName, userName, dockerLaunchCommand, configID, hostWorkingDir, configPath, outputsDirectory);
      });

      return CeresTrainCommandUtils.DoTrainingEpilogue(hostName, configID, startTime, logger, consoleStatusTable);
    }


    #region Internal helpers

    // End of training will be recognized by this string.
    const string END_TRAINING_PHRASE = "INFO: EXIT_STATUS";

    static void DoGoRemote(TrainingStatusTable table, FileLogger logger,
                           string hostName, string userName, 
                           string dockerLaunchCommand,
                           string configID, string baseDir,
                           string configBasePath, string outputsDirectory)
    {
      using SSHClient sshClientx = new SSHClient(hostName, userName);
      sshClientx.CheckSSHClientConnected();

      SshClient sshClient = sshClientx.SSHExecClient;
      sshClient.Connect();

      lock (consoleOutLockObject)
      {
        Console.WriteLine();
        Console.WriteLine("Connected to " + hostName);
      }

      int numTrainLinesSeen = 0;
      DateTime startTime = default;
      //python3 train.py /mnt/deve/cout/configs/C5_B1_512_15_16_4_32bn_2024 /mnt/deve/cout
      string command = $"cd {baseDir} && python3 train.py {configID} {outputsDirectory}";

      if (dockerLaunchCommand != null)
      {
        command = $"{dockerLaunchCommand} bash -c \"{command}\"";
      } 

      lock (consoleOutLockObject)
      {
        Console.WriteLine();
        ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"Launching on {hostName} as {userName} via SSH: {command}");
      }

      using (ShellStream shellStream = sshClient.CreateShellStream("CeresTrainPy_remote_" + Environment.MachineName, 80, 24, 800, 600, 1024))
      {
        lock (consoleOutLockObject)
        {

          // Writing command to the shell stream and executing it
          shellStream.WriteLine(command);

          // Final command to exit the shell.
          shellStream.WriteLine("exit");

          Console.WriteLine($"Beginning initialization/execution of train.py on for {configBasePath}...");
          Console.WriteLine();
        }

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
            lock (consoleOutLockObject )
            {

              if (numLinesRead == 0)
              {
                ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"Immediate error/disconnect received from host {hostName} with command {command} for {configBasePath}");
              }
              else
              {
                ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"After {numLinesRead} lines received from host {hostName} for {configBasePath}, error/disconnected received. See log file at {logger.LiveLogFileName}");
              }
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
            CeresTrainCommandUtils.UpdateTableWithLine(table, configID, hostName, ref startTime, ref numTrainLinesSeen, line);

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
