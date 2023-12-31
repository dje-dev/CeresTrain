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

using static TorchSharp.torch;

#endregion

namespace CeresTrain.Utils.TensorFileCaching
{
  /// <summary>
  /// Manages a disk file cache of sequences of pairs of tensors (training data and target data).
  /// </summary>
  public class TensorPairFileCache : IDisposable
  {
    private readonly TensorFileCache underlyingWriter1;
    private readonly TensorFileCache underlyingWriter2;

    public TensorPairFileCache(string fileName, int lengthOfEachArray1, int lengthOfEachArray2, int numArraysToBufferBeforeWriteToFile = 8)
    {
      underlyingWriter1 = new TensorFileCache(fileName + ".1", lengthOfEachArray1, numArraysToBufferBeforeWriteToFile);
      underlyingWriter2 = new TensorFileCache(fileName + ".2", lengthOfEachArray2, numArraysToBufferBeforeWriteToFile);
    }

    public void AddTensors(Tensor tensor1, Tensor tensor2)
    {
      underlyingWriter1.AddTensor(tensor1);
      underlyingWriter2.AddTensor(tensor2);
    }

    public void Dispose()
    {
      underlyingWriter1.Dispose();
      underlyingWriter2.Dispose();
    }


    /// <summary>
    /// Enumerates data in file, returning as Tensors of specified type on specified device.
    /// </summary>
    /// <param name="fileName"></param>
    /// <param name="sizeOfEachArray"></param>
    /// <param name="device"></param>
    /// <param name="type"></param>
    /// <returns></returns>
    public static IEnumerable<(Tensor, Tensor)> EnumerateTensorsInFile(string fileName, int sizeOfEachArray1, int sizeOfEachArray2, Device device, ScalarType type)
    {
      IEnumerator<Tensor> reader1 = TensorFileCache.EnumerateTensorsInFile(fileName + ".1", sizeOfEachArray1, device, type).GetEnumerator();
      IEnumerator<Tensor> reader2 = TensorFileCache.EnumerateTensorsInFile(fileName + ".2", sizeOfEachArray2, device, type).GetEnumerator();

      while (reader1.MoveNext())
      {
        reader2.MoveNext();
        yield return (reader1.Current, reader2.Current);
      }
    }
  }

}
