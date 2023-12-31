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

using System.IO;
using System.Collections.Generic;

using Xunit;

using static TorchSharp.torch;

#endregion

namespace CeresTrain.Utils.TensorFileCaching
{
  /// <summary>
  /// Unit test for class TensorPairFileCache.
  /// </summary>
  public class TensorPairFileCacheTests
  {
    [Fact]
    public static void EnumerateTensorsInFile_ShouldReturnTensorsWithCorrectDataAndDevice()
    {
      string fileName = "TensorPairFileCacheTests_test_file.dat";
      File.Delete(fileName + ".1");
      File.Delete(fileName + ".2");

      const int sizeOfEachArray1 = 512;
      const int sizeOfEachArray2 = 4;
      TensorPairFileCache tensorCache = new TensorPairFileCache(fileName, sizeOfEachArray1, sizeOfEachArray2);

      const int COUNT = 5;
      float[][] testT1 = new float[COUNT][];
      float[][] testT2 = new float[COUNT][];
      for (int i = 0; i < COUNT; i++)
      {
        Tensor t1 = rand(sizeOfEachArray1);
        Tensor t2 = rand(sizeOfEachArray2);
        testT1[i] = t1.cpu().to(ScalarType.Float32).data<float>().ToArray();
        testT2[i] = t2.cpu().to(ScalarType.Float32).data<float>().ToArray();

        tensorCache.AddTensors(t1, t2);
      }
      tensorCache.Dispose();

      Device device = new Device("cpu");
      ScalarType type = ScalarType.Float32;

      IEnumerable<(Tensor, Tensor)> tensors = TensorPairFileCache.EnumerateTensorsInFile(fileName, sizeOfEachArray1, sizeOfEachArray2, device, type);

      int count = 0;
      foreach ((Tensor tensor1, Tensor tensor2) output in tensors)
      {
        Assert.True(device.type == output.tensor1.device.type);
        Assert.True(device.type == output.tensor2.device.type);

        float[] t1 = output.tensor1.cpu().to(ScalarType.Float32).data<float>().ToArray();
        float[] t2 = output.tensor2.cpu().to(ScalarType.Float32).data<float>().ToArray();

        float EPSILON = 1E-3f;
        for (int i = 0; i < t1.Length; i++)
        {
          Assert.Equal(t1[i], testT1[count][i], EPSILON);
        }
        for (int i = 0; i < t2.Length; i++)
        {
          Assert.Equal(t2[i], testT2[count][i], EPSILON);
        }

        count++;
      }

      Assert.Equal(COUNT, count);
    }
  }

}
