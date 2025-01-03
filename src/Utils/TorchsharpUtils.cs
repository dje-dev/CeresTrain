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

using ManagedCuda;

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;

using Ceres.Base.OperatingSystem;
using CeresTrain.Networks.Transformer;
using static TorchSharp.torch;

#endregion

namespace CeresTrain.Utils
{
  /// <summary>
  /// Static helper methods related to TorchSharp.
  /// </summary>
  public static class TorchSharpUtils
  {
    /// <summary>
    /// Returns string representation of the shapes.
    /// </summary>
    /// <param name="shapes"></param>
    /// <returns></returns>
    public static string ShapeStr(long[] shapes)
    {
      string ret = "[";
      for (int i = 0; i < shapes.Length; i++)
      {
        ret += shapes[i] + (i == shapes.Length - 1 ? "" : ", ");
      }
      return ret + "]";
    }


    /// <summary>
    /// Returns number of parameters in a TorchSharp module.
    /// </summary>
    /// <param name="model"></param>
    /// <returns></returns>
    public static long NumModuleParameters(Module model, bool verbose = false)
    {
      long count = 0;
      foreach (Parameter p in model.parameters())
      {
        count += p.numel();
        if (verbose && p.dim() > 1)
        {
          Console.WriteLine(p);
        }
      }
      return count;
    }


    /// <summary>
    /// Throws an exception if the specified device is determined to be invalid for this system,
    /// otherwise returns the device passed in.
    /// </summary>
    public static torch.Device ValidatedDevice(torch.Device device)
    {
      if (device == default)
      {
        throw new ArgumentNullException(nameof(device));
      }

      string deviceString = device.ToString();
      string[] split = deviceString.Split(":");

      string deviceType = split[0];

      if (deviceType.ToUpper() == "CUDA")
      {
        if (!SoftwareManager.IsCUDAInstalled) // fails due to missing entry point: (!torch.cuda.is_available())
        {
          throw new Exception("CUDA is not available on this computer.");
        }

        int deviceIndex = 0;
        if (split.Length > 1 && !int.TryParse(split[1], out deviceIndex))
        {
          throw new Exception($"The device index '{split[1]}' is not a valid integer.");
        }

        // Check if the device index is in the range of available CUDA devices
        if (deviceIndex >= CudaContext.GetDeviceCount()) // fails due to missing entry point: (deviceIndex >= torch.cuda.device_count())
        {
          throw new Exception($"Device index {deviceIndex} is out of range. Only {torch.cuda.device_count()} CUDA device(s) available.");
        }
      }

      return device;
    }


    /// <summary>
    /// Applies the specified activation function to the input.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="activation"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public static Tensor WithActivation(Tensor x, NetTransformerDef.ActivationType activation)
    {
      return activation switch
      {
        NetTransformerDef.ActivationType.None => x,
        NetTransformerDef.ActivationType.ReLUSquared => functional.relu(x).square(),
        NetTransformerDef.ActivationType.ReLU => functional.relu(x),
        NetTransformerDef.ActivationType.Swish => x * functional.sigmoid(x),
        NetTransformerDef.ActivationType.SwiGLU => functional.silu(x), // first part of SwiGLU here
        NetTransformerDef.ActivationType.Mish => functional.Mish(x),
        _ => throw new NotImplementedException()
      };
    }

    /// <summary>
    /// Returns entropy of the specified probabilities.
    /// </summary>
    /// <param name="probabilities"></param>
    /// <returns></returns>
    public static Tensor Entropy(Tensor probabilities)
    {
      float epsilon = 1e-6f;
      Tensor clippedProbabilities = clamp(probabilities + epsilon, epsilon, 1.0f); // Ensure probabilities are in a valid range
      Tensor logProbabilities = log(clippedProbabilities); // Log of probabilities
      Tensor entropy = torch.nn.functional.cross_entropy(logProbabilities, clippedProbabilities); // Entropy as self cross-entropy
      return entropy;
    }


    /// <summary>
    /// Freezes  (disable gradient updates) all layers having names satisfying specified predicate.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="layerNameShouldFreezePredicate"></param>
    /// <param name="verbose"></param>
    public static void FreezeLayers(Module model, Predicate<string> layerNameShouldFreezePredicate, bool verbose = true)
    {
      foreach (var (name, param) in model.named_parameters())
      {
        if (layerNameShouldFreezePredicate(name))
        {
          if (verbose)
          {
            Console.WriteLine("freeze " + name);
          }

          param.requires_grad = false;
        }
      }
    }
  }

}
