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
using CeresTrain.PositionGenerators;
using CeresTrain.Trainer;
using CeresTrain.Networks.Transformer;
using CeresTrain.UserSettings;

#endregion 

namespace CeresTrain.Examples
{
  public static class TrainingHelpers
  {
    /// <summary>
    /// Writes/rewrites the current configuration to disk (as JSON files).
    /// </summary>
    /// <param name="directory"></param>
    /// <param name="configID"></param>
    public static void WriteConfigJSON(in ConfigTraining configTraining, string directory, string configID)
    {
      string baseFileName = Path.Combine(directory, configID);
      ConfigSerializationJSON.WriteConfigJSON(baseFileName, configTraining with { ExecConfig = configTraining.ExecConfig with { ID = configID } });
    }


    /// <summary>
    /// Loads a specified configuration from disk.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="deviceIDs"></param>
    public static ConfigTraining AdjustAndLoadConfig(string id, string piecesStr,
                                                     int[] deviceIDs = null, string pyTorchCompileMode = null,
                                                     long? numPos = null, string? tpgDir = null)
    {
      if (deviceIDs == null || deviceIDs.Length == 0)
      {
        deviceIDs = [0];
      }

      ConfigSerializationJSON.ReadConfigJSON(id,
                                             out ConfigData configData, 
                                             out ConfigNetExecution configExec,
                                             out NetTransformerDef configTransformerDef, 
                                             out ConfigOptimization configOptimization, 
                                             out ConfigMonitoring configMonitoring);

      configOptimization = configOptimization with
      { 
        NumTrainingPositions = numPos ?? configOptimization.NumTrainingPositions
      };

      configData = configData with
      {
        SourceType = configData.SourceType,
        PositionGenerator = piecesStr != null ? new PositionGeneratorRandomFromPieces(piecesStr) : null,
        TrainingFilesDirectory = tpgDir ?? configData.TrainingFilesDirectory,
      };

      if (deviceIDs != null)
      {
        configExec = configExec with { DeviceIDs = deviceIDs };
      }

      configOptimization = configOptimization with
      {
        PyTorchCompileMode = pyTorchCompileMode ?? configOptimization.PyTorchCompileMode,
      };

      if (piecesStr == null && configData.TrainingFilesDirectory == null)
      {
        throw new ArgumentException("Must specify either piecesStr or dataFilesDir");
      }

      if (piecesStr != null && tpgDir != null)
      {
        throw new ArgumentException("Implementation limitation: pieces filter not currently supported when reading from training data files");
      }

      ConfigSerializationJSON.WriteConfigJSON(id, configData, configExec, configTransformerDef, configOptimization, configMonitoring);

      return new ConfigTraining(configExec, configTransformerDef, configData, configOptimization, configMonitoring);
    }



  }

}
