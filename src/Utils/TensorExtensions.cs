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

using static TorchSharp.torch;

using Ceres.Base.DataType;

#endregion

namespace CeresTrain.Utils
{
  /// <summary>
  /// Static extension methods for TorchSharp Tensor class.
  /// </summary>
  public static class TensorExtensions
  {
    /// <summary>
    /// Returns a float array containing the data in the tensor.
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static float[] FloatArray(this Tensor tensor) => tensor.to(ScalarType.Float32).cpu().data<float>().ToArray();


    /// <summary>
    /// Returns a float array containing the data in the tensor, reshaped as a 2D array.
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="secondDim"></param>
    /// <returns></returns>
    public static float[,] FloatArray2D(this Tensor tensor, int secondDim) => ArrayUtils.To2D(tensor.to(ScalarType.Float32).cpu().data<float>().ToArray(), secondDim);


    /// <summary>
    /// Returns a Half array containing the data in the tensor.
    /// </summary>
    /// <param name="tensor"></param>
    /// <returns></returns>
    public static Half[] HalfArray(this Tensor tensor) => tensor.to(ScalarType.Float16).cpu().data<Half>().ToArray();


    /// <summary>
    /// Returns a Half array containing the data in the tensor, reshaped as a 2D array.
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="secondDim"></param>
    /// <returns></returns>
    public static Half[,] HalfArray2D(this Tensor tensor, int secondDim) => ArrayUtils.To2D(tensor.to(ScalarType.Float16).cpu().data<Half>().ToArray(), secondDim);


    /// <summary>
    /// Returns a float array containing the data in the tensor, reshaped as a 3D array.
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="secondDim"></param>
    /// <param name="thirdDim"></param>
    /// <returns></returns>
    public static float[,,] FloatArray3D(this Tensor tensor, int secondDim, int thirdDim) => ArrayUtils.To3D(tensor.to(ScalarType.Float32).cpu().data<float>().ToArray(), secondDim, thirdDim);


    /// <summary>
    /// Returns a Half array containing the data in the tensor, reshaped as a 3D array.
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="secondDim"></param>
    /// <param name="thirdDim"></param>
    /// <returns></returns>
    public static Half[,,] HalfArray3D(this Tensor tensor, int secondDim, int thirdDim) => ArrayUtils.To3D(tensor.to(ScalarType.Float16).cpu().data<Half>().ToArray(), secondDim, thirdDim);
  }
}
