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
using System.Collections.Generic;
using System.Threading.Tasks;

using static TorchSharp.torch;

#endregion

namespace CeresTrain.Utils.TensorFileCaching
{
  /// <summary>
  /// Manages reading and writing from files which contain Tensor data (stored as Half).
  /// </summary>
  public class TensorFileCache : IDisposable
  {
    private readonly ArrayFileCache<Half> underlyingWriter;
    Half[] arrayHTemp;


    public TensorFileCache(string fileName, int lengthOfEachArray, int numArraysToBufferBeforeWriteToFile = 8)
    {
      underlyingWriter = new ArrayFileCache<Half>(fileName, lengthOfEachArray, numArraysToBufferBeforeWriteToFile);
      arrayHTemp = new Half[lengthOfEachArray];
    }

    public void AddTensor(Tensor tensor)
    {
      // Retrieve from GPU
      float[] array = tensor.cpu().to(ScalarType.Float32).data<float>().ToArray();

      Parallel.For(0, array.Length, (i) => { arrayHTemp[i] = (Half)array[i]; });

      underlyingWriter.AddArray(arrayHTemp);
    }

    public void Dispose()
    {
      underlyingWriter.Dispose();
    }


    /// <summary>
    /// Enumerates data in file, returning as Tensors of specified type on specified device.
    /// </summary>
    /// <param name="fileName"></param>
    /// <param name="sizeOfEachArray"></param>
    /// <param name="device"></param>
    /// <param name="type"></param>
    /// <returns></returns>
    public static IEnumerable<Tensor> EnumerateTensorsInFile(string fileName, int sizeOfEachArray, Device device, ScalarType type)
    {
      if (new FileInfo(fileName).Length % sizeOfEachArray != 0)
      {
        throw new Exception($"Length of file {fileName} is not multiple of {nameof(sizeOfEachArray)}");
      }

      // TODO: Find a way to do this on GPU/Torchsharp instead?
      foreach (Half[] halfArray in ArrayFileCache<Half>.EnumerateArraysInFile<Half>(fileName, sizeOfEachArray))
      {
        float[] floats = new float[halfArray.Length];

        for (int i = 0; i < halfArray.Length / 2; i++)
        {
          int offset = i * 2;
          floats[offset] = (float)halfArray[offset];
          floats[offset + 1] = (float)halfArray[offset + 1];
        }

        // Handle leftover, if any.
        if (floats.Length % 2 != 0)
        {
          floats[floats.Length - 1] = (float)halfArray[floats.Length - 1];
        }

        yield return from_array(floats, device: device).to(type);
      }
    }
  }

}
