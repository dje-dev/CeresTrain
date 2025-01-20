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

using Ceres.Chess.NNEvaluators.Ceres.TPG;
using CeresTrain.TPG;
using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;


#endregion

namespace CeresTrain.TPGDatasets
{
  /// <summary>
  /// Sublcass of Torch's Dataset to provide raw training data sourced from TPG files.
  /// </summary>
  public sealed class TorchDatasetFromTPG : Dataset
  {
    public override long Count => long.MaxValue;

    TorchDatasetFromTPGWorker[] subsets;

    int nextIndex = 0;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="tpgOrTARSourceDirectory"></param>
    /// <param name="fractionQ"></param>
    /// <param name="device"></param>
    /// <param name="dataType"></param>
    /// <param name="batchSize"></param>
    /// <param name="wdlLabelSmoothing"></param>
    /// <param name="overrideRecordEnumerator"></param>
    /// <param name="countParallel"></param>
    public TorchDatasetFromTPG(string tpgOrTARSourceDirectory,
                               float fractionQ,
                               Device device, ScalarType dataType, int batchSize,
                               float wdlLabelSmoothing,
                               IEnumerator<TPGRecord[]> overrideRecordEnumerator,
                               int countParallel = 3)
    {
      // Throw if tpgDirectory does not exist
      if (tpgOrTARSourceDirectory != null && !System.IO.Directory.Exists(tpgOrTARSourceDirectory))
      {
        throw new System.IO.DirectoryNotFoundException($"Directory {tpgOrTARSourceDirectory} does not exist");
      }

#if NOT
      int numSourceFiles = overrideRecordEnumerator == null ? System.IO.Directory.GetFiles(tpgOrTARSourceDirectory, "*.zst").Length 
                                                            : System.IO.Directory.GetFiles(tpgOrTARSourceDirectory, "*.tar").Length;
      // Verify that there are some files in the directory.
      if (numSourceFiles == 0)
      {
        throw new System.IO.DirectoryNotFoundException($"Directory {tpgOrTARSourceDirectory} " 
          + $"contains no {(overrideRecordEnumerator == null ? ".zst" : ".tar")} files");
      }

      // Don't start more workers than there are files!
      countParallel = System.Math.Min(countParallel, numSourceFiles);

      // Also avoid situation where some workers would have 2 files and others only 1.
      if (numSourceFiles < 2 * countParallel)
      {
        countParallel = numSourceFiles / 2;
      }
#endif

 countParallel = 1; // Dataset can't be parallelized due to CUDA overlap operations (?)

      subsets = new TorchDatasetFromTPGWorker[countParallel];
      for (int i = 0; i < countParallel; i++)
      {
        subsets[i] = new TorchDatasetFromTPGWorker(tpgOrTARSourceDirectory, countParallel, i,
                                                   fractionQ, device, dataType, batchSize, 
                                                   wdlLabelSmoothing, overrideRecordEnumerator);
      }
    }


    public (Dictionary<string, Tensor> dict, TPGRecord[] rawRecords) GetTensorAndRawRecords(long index, ScalarType dataType)
    {
      // Cycle thru the subsets.
      return subsets[nextIndex++ % subsets.Length].GetTensorAndRawRecords(index, dataType);
    }


    public override Dictionary<string, Tensor> GetTensor(long index)
    {
      // Cycle thru the subsets.
      return subsets[nextIndex++ % subsets.Length].GetTensor(index);
    }
  }
}
