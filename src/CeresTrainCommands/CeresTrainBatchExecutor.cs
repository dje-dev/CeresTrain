﻿#region License notice

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
    public static TrainingResultSummary[] TestBatchParallel(string piecesString, long numPositions,
                                                            params (string variantID, CeresTrainHostConfig hostConfig, int[] deviceIDs, string tpgDir, ConfigTraining baseConfig, Func<ConfigTraining, ConfigTraining> modifier)[] variants)
    {
      // Throw exception if any of the variants have the same ID
      string[] variantsArray = variants.Select(v => v.variantID).ToArray();
      if (variantsArray.Distinct().Count() != variantsArray.Length)
      {
        throw new Exception("Duplicate variant IDs in variants array");
      }

      CeresTrainInitialization.InitializeCeresTrainEnvironment();

      TrainingStatusTable sharedStatusTable = new TrainingStatusTable("SHARED TRAIN", "SHARED TRAIN", numPositions, true);

      List<TrainingResultSummary> results = new();

      object lockObj = new();

      string configsDir = CeresTrainUserSettingsManager.Settings.OutputsDir + "/configs";
      Parallel.ForEach(variants, variant =>
      {
        string fullConfigID = variant.baseConfig.ExecConfig.ID + "_" + variant.variantID;
        ConfigTraining config = variant.baseConfig;
        config = variant.modifier(config);
        CeresTrainCommands.ProcessInitCommand(in config, fullConfigID);

        lock (lockObj)
        {
          Console.WriteLine();
          Console.WriteLine("Writing config " + fullConfigID);
          Console.WriteLine();
          Console.WriteLine("Launch " + fullConfigID + " on " + variant.hostConfig.HostName + " ...");
        }

        TrainingResultSummary result = CeresTrainCommands.ProcessTrainCommand(fullConfigID, piecesString, numPositions, variant.hostConfig.HostName, variant.tpgDir, variant.deviceIDs, sharedStatusTable);

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
}
