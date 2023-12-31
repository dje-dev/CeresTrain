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
using System.Text.Json.Serialization;
using System.Text.Json;

using CeresTrain.Networks.Transformer;
using CeresTrain.UserSettings;
using CeresTrain.Utils;
using Ceres.Chess.SearchResultVerboseMoveInfo;


#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Helper class for writing JSON files with training configuration information.
  /// These files can be used for:
  ///   - saving a record of the training configuration used for a particular training run
  ///   - transferring a training configuration from one machine to another
  ///   - transferring a training configuration to another backend (such as the PyTorch/Python backend)
  /// </summary>
  public static class ConfigSerializationJSON
  {
    /// <summary>
    /// Reads 5 JSON files containing the configuration parameters used for training
    /// and combines them into a ConfigTraining object.
    /// </summary>
    /// <param name="baseOutputPath"></param>
    /// <returns></returns>
    public static ConfigTraining ReadConfigJSON(string baseOutputPath)
    {
      ReadConfigJSON(baseOutputPath,
                     out ConfigData dataConfig,
                     out ConfigNetExecution execConfig,
                     out NetTransformerDef netConfig,
                     out ConfigOptimization optConfig,
                     out ConfigMonitoring monitoringConfig);
      return new ConfigTraining(execConfig, netConfig, dataConfig, optConfig, monitoringConfig);      
    }

    /// <summary>
    /// Emits 5 JSON files containing the ConfigTraining configuration information.
    /// </summary>
    /// <param name="baseFileName"></param>
    /// <param name="config"></param>
    public static void WriteConfigJSON(string baseFileName, in ConfigTraining config)
    {
      if (config.NetDefConfig is not NetTransformerDef)
      {
        throw new Exception("Specified ConfigTraining did not contain a NetTransformerDef for the NetDefConfig. Currently only Transformer networks are supported.");
      }

      WriteConfigJSON(baseFileName, config.DataConfig, config.ExecConfig, 
                      (NetTransformerDef)config.NetDefConfig,
                      config.OptConfig, config.MonitoringConfig);                             
    } 

    /// <summary>
    /// Reads 5 JSON files containing the configuration parameters used for training.
    /// </summary>
    /// <param name="baseOutputPath"></param>
    /// <param name="id"></param>
    /// <param name="dataConfig"></param>
    /// <param name="execConfig"></param>
    /// <param name="netConfig"></param>
    /// <param name="optConfig"></param>
    /// <param name="monitoringConfig"></param>
    public static void ReadConfigJSON(string baseConfigFileName,
                                      out ConfigData dataConfig,
                                      out ConfigNetExecution execConfig,
                                      out NetTransformerDef netConfig,
                                      out ConfigOptimization optConfig,
                                      out ConfigMonitoring monitoringConfig)
    {
      baseConfigFileName = baseConfigFileName + "_" + "ceres_";

      JsonSerializerOptions options = new JsonSerializerOptions
      {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() }
      };

      dataConfig = JsonSerializer.Deserialize<ConfigData>(File.ReadAllText(baseConfigFileName + "data.json"), options);
      execConfig = JsonSerializer.Deserialize<ConfigNetExecution>(File.ReadAllText(baseConfigFileName + "exec.json"), options);
      netConfig = JsonSerializer.Deserialize<NetTransformerDef>(File.ReadAllText(baseConfigFileName + "net.json"), options);
      optConfig = JsonSerializer.Deserialize<ConfigOptimization>(File.ReadAllText(baseConfigFileName +  "opt.json"), options);
      monitoringConfig = JsonSerializer.Deserialize<ConfigMonitoring>(File.ReadAllText(baseConfigFileName + "monitoring.json"), options);
    }


    /// <summary>
    /// Emits 5 JSON files containing the configuration parameters used for training.
    /// </summary>
    /// <param name="baseFileName"></param>
    /// <param name="dataConfig"></param>
    /// <param name="execConfig"></param>
    /// <param name="netConfig"></param>
    /// <param name="optConfig"></param>
    public static void WriteConfigJSON(string baseFileName,
                                       in ConfigData dataConfig,
                                       in ConfigNetExecution execConfig,
                                       in NetTransformerDef netConfig,
                                       in ConfigOptimization optConfig,
                                       in ConfigMonitoring monitoringConfig)
    {
      baseFileName = Path.Combine(baseFileName, baseFileName + "_" + "ceres_");
      JsonSerializerOptions options = new JsonSerializerOptions
      {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() }
      };

      File.WriteAllText(baseFileName + "exec.json", JsonSerializer.Serialize(execConfig, options));
      File.WriteAllText(baseFileName + "net.json", JsonSerializer.Serialize(netConfig, options));
      File.WriteAllText(baseFileName + "opt.json", JsonSerializer.Serialize(optConfig, options));
      File.WriteAllText(baseFileName + "data.json", JsonSerializer.Serialize(dataConfig, options));
      File.WriteAllText(baseFileName + "monitoring.json", JsonSerializer.Serialize(monitoringConfig, options));
    }


    /// <summary>
    /// Uploads a configuration from a local source directory to a remote target directory.
    /// </summary>
    /// <param name="sshConnection"></param>
    /// <param name="configName"></param>
    /// <param name="sourceDir"></param>
    /// <param name="targetDir"></param>
    /// <param name="verbose"></param>
    public static void UploadConfig(SSHClient sshConnection, string configName, string sourceDir, string targetDir, bool verbose = true)
    {
      string[] files = new string[] { "ceres_data.json", "ceres_exec.json", "ceres_monitoring.json", "ceres_net.json", "ceres_opt.json" };
      foreach (string file in files)
      {
        string sourceFN = Path.Combine(sourceDir, configName + "_" + file);
        string targetFN = targetDir + "/" +  configName + "_" + file;

        sshConnection.UploadFile(sourceFN, targetFN,  logToConsole:verbose);
      }
    }     


    /// <summary>
    /// Dumps the contents of the 5 JSON files to the console.
    /// </summary>
    /// <param name="directory"></param>
    /// <param name="configName"></param>
    /// <param name="configType"></param>
    static void Dump(string directory, string configName, string configType)
    {
      Console.WriteLine();
      string fn = Path.Combine(directory, $"{configName}_ceres_{configType}.json");

      // Write file name followed by lines in file (intended).
      Console.WriteLine(fn);
      foreach (string line in File.ReadAllLines(fn))
      {
        Console.WriteLine("  " + line);
      }
    }


    /// <summary>
    /// Dumps the contents of the 5 JSON files to the console for a given configuration in default configs directory.
    /// </summary>
    /// <param name="configName"></param>
    public static void DumpJSONToConsole(string configName) =>
       DumpJSONToConsole(Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "configs"), configName); 
    

    /// <summary>
    /// Dumps the contents of the 5 JSON files to the console for a given configuration in a given directory.
    /// </summary>
    /// <param name="directory"></param>
    /// <param name="configName"></param>
    public static void DumpJSONToConsole(string directory, string configName)
    {
      Dump(directory, configName, "exec");
      Dump(directory, configName, "net");
      Dump(directory, configName, "opt");
      Dump(directory, configName, "data");
      Dump(directory, configName, "monitoring");
    }

  }
}
