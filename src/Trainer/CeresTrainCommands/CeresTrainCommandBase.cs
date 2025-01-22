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

using TorchSharp;
using static TorchSharp.torch;

using CeresTrain.Utils;
using CeresTrain.Networks;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Base class for all CeresTrain commands.
  /// </summary>
  public abstract class CeresTrainCommandBase
  {
    /// <summary>
    /// Training configuration to be used by commands.
    /// </summary>
    public ConfigTraining TrainingConfig;

    /// <summary>
    /// Neural network model being used.
    /// </summary>
    public CeresNeuralNet Model;

    /// <summary>
    /// Convenience accessor to the optimization batch size (for each forward pass).
    /// </summary>
    public int OptimizationBatchSizeForward => TrainingConfig.OptConfig.BatchSizeForwardPass;

    /// <summary>
    /// Convenience accessor to the optimization batch size (for the backward step with possibly accumulated gradients).
    /// </summary>
    public int OptimizationBatchSizeBackward => TrainingConfig.OptConfig.BatchSizeBackwardPass;


    /// <summary>
    /// Base class constructor.
    /// </summary>
    /// <param name="trainingConfig"></param>
    public CeresTrainCommandBase(in ConfigTraining trainingConfig)
    {
      TrainingConfig = trainingConfig;
    }


    #region Helper methods

    /// <summary>
    /// Initializes the trainer before any commands performed.
    /// </summary>
    /// <exception cref="Exception"></exception>
    public void PrepareTrainer()
    {
      if (Model == null) // if not already prepared
      {
        Console.WriteLine(" PyTorch version  : " + __version__);
        Console.WriteLine("  Training device : " + TrainingConfig.ExecConfig.Device);

        // Console.WriteLine("EXPORT model as described here https://github.com/microsoft/Windows-Machine-Learning/blob/65a3ce340d34e7d9a6629ccab4a6da4a1722c258/Samples/Tutorial%20Samples/PyTorch%20Data%20Analysis/PyTorch%20Training%20-%20Data%20Analysis/DataClassifier.py#L63");

        // Possibly load initial weights.
        string fnStartWeights = null;
        Dictionary<string, Tensor> weightsStart = null;
        if (fnStartWeights != null)
        {
          var transformerTS = TorchscriptUtils.TorchScriptFilesAveraged<Tensor, Tensor, (Tensor, Tensor, Tensor, Tensor)>
            (TrainingConfig.ExecConfig.SaveNetwork1FileName, TrainingConfig.ExecConfig.SaveNetwork1FileName,
             TrainingConfig.ExecConfig.Device, TrainingConfig.ExecConfig.DataType);

          weightsStart = new();
          transformerTS.named_parameters().ToList().ForEach(p => weightsStart.Add(p.name, p.parameter.AsParameter().detach()));
        }

        Model = TrainingConfig.NetDefConfig.CreateNetwork(TrainingConfig.ExecConfig);
        if (weightsStart != null)
        {
          throw new Exception("Not yet implemented: passing in weights");
        }
      }
    }


    #endregion
  }
}
