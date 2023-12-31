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
    public static TrainingResultSummary[] TestBatch(in ConfigTraining baseConfig, string piecesString,
                                                    in CeresTrainHostConfig hostConfig,
                                                    string tpgDir, int[] deviceIDs, long numPositions,
                                                    params (string variantID, Func<ConfigTraining, ConfigTraining> modifier)[] variantModifiers)
    {
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
        TrainingResultSummary result = CeresTrainCommands.ProcessTrainCommand(fullConfigID, piecesString, numPositions, hostConfig.HostName, tpgDir, deviceIDs);
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
