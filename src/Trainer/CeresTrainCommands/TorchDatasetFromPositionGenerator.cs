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
using System.Threading;
using System.Collections.Generic;
using System.Threading.Tasks;

using static TorchSharp.torch.utils.data;
using static TorchSharp.torch;

using Ceres.Base.Misc;
using CeresTrain.PositionGenerators;

using CeresTrain.TPG;
using CeresTrain.TPGDatasets;
using CeresTrain.TrainingDataGenerator;
using Ceres.Chess.NNEvaluators.Ceres.TPG;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Subclass of Torchsharp Dataset which returns dataset items
  /// derived from a specified generator function (which returns a Position).
  /// </summary>
  public class TorchDatasetFromPositionGenerator : Dataset
  {
    /// <summary>
    /// Data type of tensors returned.
    /// </summary>
    public readonly ScalarType DataType;

    /// <summary>
    /// Number of positions in each batch.
    /// </summary>
    public readonly int BatchSize;

    /// <summary>
    /// Returns total number of batches that can be generated (assume infinite).
    /// </summary>
    public override long Count => long.MaxValue;


    TPGBatchToTensorDictConverter dictConverter;
    TablebaseTPGBatchGenerator generator;

    // TODO: implement graceful shutdown


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="posGenerator"></param>
    /// <param name="succeedIfIncompleteDTZ"></param>
    /// <param name="device"></param>
    /// <param name="dataType"></param>
    /// <param name="batchSize"></param>
    /// <param name="wdlLabelSmoothing"></param>
    /// <param name="numWorkerThreads"></param>
    /// <param name="maxBatchQueueLength"></param>
    public TorchDatasetFromPositionGenerator(PositionGenerator posGenerator, bool succeedIfIncompleteDTZ,
                                             ScalarType dataType, Device device,
                                             int batchSize, float wdlLabelSmoothing, int numWorkerThreads = 4, int maxBatchQueueLength = 2)
    {
      BatchSize = batchSize;
      DataType = dataType;

      dictConverter = new TPGBatchToTensorDictConverter(0, device, dataType, batchSize, wdlLabelSmoothing);
      generator = new TablebaseTPGBatchGenerator(posGenerator.ID, posGenerator.GeneratePosition, succeedIfIncompleteDTZ, batchSize, numWorkerThreads);

      // Start the generator on a background thread.
      Task generatorTask = new Task(() =>
      {
        try
        {
          generator.Start(maxBatchQueueLength);
        }
        catch (Exception e)
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Exception in TorchDatasetFromPositionGenerator, abort.");
          Console.WriteLine(e);
          Environment.Exit(3);
        }
      });
      generatorTask.Start();
    }


    /// <summary>
    /// Returns next batch of TPGRecord[].
    /// </summary>
    /// <returns></returns>
    public TPGRecord[] GetBatch()
    {
      TPGRecord[] batch;
      while (!generator.PendingRecords.TryDequeue(out batch))
      {
        Thread.Sleep(30);
      }
      return batch;
    }


    /// <summary>
    /// Overridden method to return a batch of tensors.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public override Dictionary<string, Tensor> GetTensor(long index)
    {
      TPGRecord[] batch = GetBatch();
      Dictionary<string, Tensor> ret = dictConverter.BuildTensorDictFromTPGRecords(batch, true);
      return ret;
    }
  }
}
