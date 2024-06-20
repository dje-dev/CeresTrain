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
using System.Linq;
using System.Threading.Tasks;
using CeresTrain.Trainer;
using CeresTrain.UserSettings;

#endregion

namespace CeresTrain.TrainCommands
{
  /// <summary>
  /// Static helper class for coordinating successive execution of multiple training sessions,
  /// each with individually modified configurations.
  /// </summary>
  public static class CeresTrainBatchExecutor
  {
    public enum BatchExecutorMode
    {
      WriteConfigsOnly,
      WriteConfigsAndRunTrainingSessions
    }



    /// <summary>
    /// Runs a series of training configurations on specified host/device.
    /// </summary>
    /// <param name="mode"></param>
    /// <param name="variants"></param>
    /// <returns></returns>
    public static TrainingResultSummary[] RunSessions(BatchExecutorMode mode, params TrainingSessionSpecification[] variants)
    {
      return RunSessions(null, mode, variants);
    }

    /// <summary>
    /// Runs a series of training configurations on specified host/device.
    /// </summary>
    /// <param name="piecesString"></param>
    /// <param name="variants"></param>
    /// <returns></returns>
    public static TrainingResultSummary[] RunSessions(string piecesString, TrainingSessionSpecification[] variants)
    {
      return RunSessions(piecesString, BatchExecutorMode.WriteConfigsAndRunTrainingSessions, variants);
    }


    /// <summary>
    /// Runs a series of training configurations on specified host/device.
    /// </summary>
    /// <param name="piecesString"></param>
    /// <param name="numPositions"></param>
    /// <param name="variants"></param>
    /// <returns></returns>
    public static TrainingResultSummary[] RunSessions(string piecesString, BatchExecutorMode mode, TrainingSessionSpecification[] variants)
    {
      // Throw exception if any of the variants have the same ID
      string[] variantsArray = variants.Select(v => v.variantID).ToArray();
      if (variantsArray.Distinct().Count() != variantsArray.Length)
      {
        throw new Exception("Duplicate variant IDs in variants array");
      }

      long maxNumPositions = variants.Max(v => v.baseConfig.OptConfig.NumTrainingPositions);
      
      CeresTrainInitialization.InitializeCeresTrainEnvironment();

      TrainingStatusTable sharedStatusTable = new TrainingStatusTable("SHARED TRAIN", "SHARED TRAIN", maxNumPositions, true);

      List<TrainingResultSummary> results = new();

      object lockObj = new();

      string configsDir = CeresTrainUserSettingsManager.Settings.OutputsDir + "/configs";
      Parallel.ForEach(variants, variant =>
      {
        string fullConfigID = variant.baseConfig.ExecConfig.ID + "_" + variant.variantID;
        ConfigTraining config = variant.baseConfig;

        lock (lockObj)
        {
          CeresTrainCommands.ProcessInitCommand(in config, fullConfigID);
          Console.WriteLine();
          Console.WriteLine("Writing config " + fullConfigID);
          Console.WriteLine();
          Console.WriteLine("Launch " + fullConfigID + " on " + variant.hostConfig.HostName + " ...");
        }

        // Random sleep to avoid overloading the server.
        System.Threading.Thread.Sleep((int)(2000f * Random.Shared.NextSingle()));

        TrainingResultSummary result = CeresTrainCommands.ProcessTrainCommand(fullConfigID, piecesString, null, variant.hostConfig.HostName, null, variant.deviceIDs, sharedStatusTable);

        lock (lockObj)
        {
          results.Add(result);
          Console.WriteLine("Done execution " + fullConfigID);
          Console.WriteLine(result.WriteJSON(configsDir, fullConfigID));
        } 
      });

      Console.WriteLine();
      foreach (TrainingResultSummary result in results)
      {
        Console.WriteLine(result.ConfigID + " " + result.LossSummary.ValueAccuracy + " " + result.LossSummary.PolicyLoss + " " + result.LossSummary.PolicyAccuracy);
      }

      return results.ToArray();
    }


    /// <summary>
    /// Runs a series of training configurations on specified host/device.
    /// </summary>
    /// <param name="baseConfig"></param>
    /// <param name="piecesString"></param>
    /// <param name="hostConfig"></param>
    /// <param name="deviceIDs"></param>
    /// <param name="numPositions"></param>
    /// <param name="variantModifiers"></param>
    /// <returns></returns>
    public static TrainingResultSummary[] TestBatchSerial(in ConfigTraining baseConfig, string piecesString,
                                                          in CeresTrainHostConfig hostConfig,
                                                          string tpgDir, int[] deviceIDs, long numPositions,
                                                          params (string variantID, Func<ConfigTraining, ConfigTraining> modifier)[] variantModifiers)
    {
      // Throw exception if any of the variants have the same ID
      string[] variantsArray = variantModifiers.Select(v => v.variantID).ToArray();
      if (variantsArray.Distinct().Count() != variantsArray.Length)
      {
        throw new Exception("Duplicate variant IDs in variants array");
      }

      CeresTrainInitialization.InitializeCeresTrainEnvironment();

      List<TrainingResultSummary> results = new();

      string configsDir = CeresTrainUserSettingsManager.Settings.OutputsDir + "/configs";
      foreach ((string variantID, Func<ConfigTraining, ConfigTraining> modifier) variant in variantModifiers)
      {
        string fullConfigID = baseConfig.ExecConfig.ID + "_" + variant.variantID;

        ConfigTraining config = baseConfig;
        config = variant.modifier(config);
        CeresTrainCommands.ProcessInitCommand(in config, fullConfigID);

        Console.WriteLine();
        Console.WriteLine("Writing config " + fullConfigID);

        Console.WriteLine();
        Console.WriteLine("Launch " + fullConfigID + " on " + hostConfig.HostName + " ...");
        TrainingStatusTable sharedStatusTable = new TrainingStatusTable(fullConfigID, fullConfigID, numPositions, true);
        TrainingResultSummary result = CeresTrainCommands.ProcessTrainCommand(fullConfigID, piecesString, numPositions, hostConfig.HostName, tpgDir, deviceIDs, sharedStatusTable);
        results.Add(result);
        Console.WriteLine("Done execution " + fullConfigID);
        Console.WriteLine(result.WriteJSON(configsDir, fullConfigID));
      }

      Console.WriteLine();
      foreach (TrainingResultSummary result in results)
      {
        Console.WriteLine(result.ConfigID + " " + result.LossSummary.ValueAccuracy + " " + result.LossSummary.PolicyLoss + " " + result.LossSummary.PolicyAccuracy);
      }

      return results.ToArray();
    }

  }

  public record struct TrainingSessionSpecification(string variantID, CeresTrainHostConfig hostConfig, int[] deviceIDs, ConfigTraining baseConfig)
  {
    public static implicit operator (string variantID, CeresTrainHostConfig hostConfig, int[] deviceIDs, ConfigTraining baseConfig)(TrainingSessionSpecification value)
    {
      return (value.variantID, value.hostConfig, value.deviceIDs, value.baseConfig);
    }

    public static implicit operator TrainingSessionSpecification((string variantID, CeresTrainHostConfig hostConfig, int[] deviceIDs, ConfigTraining baseConfig) value)
    {
      return new TrainingSessionSpecification(value.variantID, value.hostConfig, value.deviceIDs, value.baseConfig);
    }
  }
}
