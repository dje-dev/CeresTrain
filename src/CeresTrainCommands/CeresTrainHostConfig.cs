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
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Ceres.Base.Misc;
using CeresTrain.UserSettings;

#endregion

namespace CeresTrain.TrainCommands
{
  /// <summary>
  /// Set of configuration settings which describe a training host on which CeresTrain commands can be executed,
  /// of one of three possible types:
  ///   - "inprocess" for local C# execution
  ///   - "WSL" for local Windows Subsystem for Linux (WSL) execution
  ///   - "<hostname>" where <hostname> is the name of a remote host on which to execute
  /// </summary>
  /// <param name="HostName">name of machine on which to execute, or "inprocess" for local C# execution, or "WSL" for local WSL</param>
  /// <param name="UserName">name of user account on host</param>
  /// <param name="CeresTrainPyDir">directory containing CeresTrainPy Python code for training</param>
  /// <param name="PathToOutputFromHost">path to director to which output files should be written</param>
  /// <param name="OverridePyTorchCompileMode">optional override mode to use for the PyTorch compile (or null for no compile)</param>
  /// <param name="DockerLaunchCommand">optional launch command for Docker is to be used</param>
  public readonly record struct CeresTrainHostConfig(string HostName, // case sensitive
                                                     string UserName,
                                                     string CeresTrainPyDir,
                                                     string PathToOutputFromHost,
                                                     string OverridePyTorchCompileMode = null,
                                                     string DockerLaunchCommand = null)
  {

    /// <summary>
    /// Set of configured hosts for training that can be populated at runtime.
    /// </summary>
    public static Dictionary<string, CeresTrainHostConfig> RegisteredConfigs = new()
    {
    };


    /// <summary>
    /// Reads a set of host configurations from a file and installs as the CeresTrainHostConfig.RegisteredConfigs.
    /// </summary>
    /// <param name="filePath"></param>
    public static void SetRegisteredHostConfigsFromFile(string filePath) 
      => RegisteredConfigs = ReadHostConfigsFromFile(filePath).ToDictionary(c => c.HostName);



    /// <summary>
    /// Writes specified IEnumerable of CeresTrainHostConfig to a file as JSON.
    /// </summary>
    /// <param name="configs"></param>
    /// <param name="filePath"></param>
    public static void WriteHostConfigsToFile(IEnumerable<CeresTrainHostConfig> configs, string filePath)
    {
      string json = JsonSerializer.Serialize(configs, new JsonSerializerOptions { WriteIndented = true });
      File.WriteAllText(filePath, json);
    }


    /// <summary>
    /// Reads a set of host configurations from a file and returns as a List of CeresTrainHostConfig.
    /// </summary>
    /// <param name="filePath"></param>
    /// <returns></returns>
    public static List<CeresTrainHostConfig> ReadHostConfigsFromFile(string filePath)
    {
      try
      {
        // Read the file content
        string json = File.ReadAllText(filePath);

        // Deserialize the JSON content into a List of CeresTrainHostConfig
        var configList = JsonSerializer.Deserialize<List<CeresTrainHostConfig>>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        return configList ?? new List<CeresTrainHostConfig>();
      }
      catch (Exception ex)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"Error reading or deserializing the file: {ex.Message}");
        return new List<CeresTrainHostConfig>();
      }
    }
  }
}
