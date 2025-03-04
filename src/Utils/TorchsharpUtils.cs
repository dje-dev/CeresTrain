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
using System.Linq;
using System.Collections.Generic;
using CeresTrain.Networks.MiscModules;
using TorchSharp.Utils;
using Ceres.Base.Misc;

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
      Tensor entropy = functional.cross_entropy(logProbabilities, clippedProbabilities); // Entropy as self cross-entropy
      return entropy;
    }

    /// <summary>
    /// Returns entropy of the specified logits.
    /// </summary>
    /// <param name="logits"></param>
    /// <returns></returns>
    public static Tensor EntropyLogits(Tensor logits) => functional.cross_entropy(logits, logits);


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


    #region Hooks

    /// <summary>
    /// Registers a forward hook on all modules in the specified root module.
    /// </summary>
    /// <param name="rootModule"></param>
    /// <param name="hookFunc"></param>
    /// <exception cref="Exception"></exception>
    public static void RegisterHooks(Module rootModule, Func<string, Module, Tensor, Tensor, Tensor, Tensor> hookFunc)
    {
      foreach ((string name, Module module) node in rootModule.named_modules())
      {
        // Explicitly handle the two common cases of one input tensor and one or two output tensors.
        if (node.module is Module<Tensor, Tensor> module)
        {
          (node.module as Module<Tensor, Tensor>).register_forward_hook((mod, input, output) =>
          {
            return hookFunc(node.name, mod, input, output, default);
          });
        }
        else if (node.module is Module<Tensor, Tensor, Tensor> module1)
        {
          (node.module as Module<Tensor, Tensor, Tensor>).register_forward_hook((mod, input, output1, output2) =>
          {
            return hookFunc(node.name, mod, input, output1, output1);
          });
        }
        else
        {
          // Ignore, possibly just an activation layer 
          Console.WriteLine("Layer " + node.name + " not hooked, not supported.");  
        }
      }
    }

    #endregion


    #region Extracting parameters for Modules

      /// <summary>
      /// Extracts weights and biases from all Linear layers in the module.
      /// 
      /// NOTE: it is assumed that the provided layers are of the same number and 
      ///       in the same order as an enumeration of the module will provide.
      /// </summary>
      /// <param name="module"></param>
      /// <param name="linearLayers"></param>
      /// <returns></returns>
      /// <exception cref="ArgumentException"></exception>
    public static Dictionary<string, (float[] weights, float[] bias)>
  ExtractWeightsFromLinearLayers(Module<Tensor, Tensor> module, params (string name, bool transpose)[] linearLayers)
    {
      int nextNameIndex = 0;

      Dictionary<string, (float[] weights, float[] bias)> ret = new();
      foreach ((string name, Module module) nn in module.named_modules())
      {
        if (nn.module.GetType() == typeof(Linear))
        {
          if (nextNameIndex >= linearLayers.Length)
          {
            throw new Exception("More Linear layers found than were provided in layers argument.");
          }

          (float[] weights, float[] bias) = ExtractWeightsAndBiasesFromModule(nn.module, linearLayers[nextNameIndex].transpose);
          ret[linearLayers[nextNameIndex++].name] = (weights, bias);
        }
        else if (nn.module.named_parameters().Count() > 0)
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "not emitting " + nn.name);
//          throw new ArgumentException("Found Module with parameters but not Linear, not supported.");
        }
      }

      return ret;
    }


    /// <summary>
    /// Extracts weights and biases from a module.
    /// 
    /// NOTE: currently only supports Linear layers.
    /// </summary>
    /// <param name="module"></param>
    /// <param name="transpose"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static (float[] weights, float[] bias) ExtractWeightsAndBiasesFromModule(Module module, bool transpose)
    {
      if (module.GetType() != typeof(Linear))
      {
        throw new ArgumentException("Module must be a Linear layer.");
      }

      Parameter weightsParam = module.named_parameters().FirstOrDefault(np => np.name.Contains("weight")).parameter;
      Parameter biasParam = module.named_parameters().FirstOrDefault(np => np.name.Contains("bias")).parameter;

      // Move to CPU, convert to float[], etc.
      float[] weights = weightsParam.cpu().to(ScalarType.Float32).data<float>().ToArray();
      float[] bias = biasParam.cpu().to(ScalarType.Float32).data<float>().ToArray();

      if (transpose)
      {
        int rows = (int)weightsParam.shape[0];
        int cols = (int)weightsParam.shape[1];

        float[] weightsTranspose = new float[rows * cols];
        for (int row = 0; row < rows; row++)
        {
          for (int col = 0; col < cols; col++)
          {
            weightsTranspose[col * rows + row] = weights[row * cols + col];
          }
        }
        weights = weightsTranspose;
      }


      return (weights, bias);
    }

    #endregion


    /// <summary>
    /// TorchSharp module that implements TeLU activation function.
    /// See: "TeLU Activation Function for Fast and Stable Deep Learning" 
    ///       by Fernandez et. al. (https://arxiv.org/abs/2412.20269).
    /// </summary>
    public class TeLU : Module<Tensor, Tensor>
    {
      public TeLU() : base("TeLU") { }

      public override Tensor forward(Tensor input)
      {
        return input * torch.tanh(torch.exp(input));
      }
    }


    /// <summary>
    /// TorchSharp module to reshape a tensor.
    /// Useful to include as part of a Sequential module.
    /// </summary>
    public class Reshape : Module<Tensor, Tensor>
    {
      private readonly long[] _shape;

      public Reshape(params long[] shape) : base("Reshape")
      {
        _shape = shape;
      }

      public override Tensor forward(Tensor input)
      {
        return input.view(_shape);
      }
    }

    /// <summary>
    /// Clips the input tensor values to the specified ranges (eliminates subnormals or excessively large values).
    /// Positive values are clamped to [minPositiveVal, maxFiniteVal] and 
    /// negative values to [-maxFiniteVal, -minPositiveVal].
    /// 
    /// This logic is similar to the ONNXRuntime clipping logic used for robustness with reduced precision.
    /// see: https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/float16.py.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="minPositiveVal">Minimum allowed positive value (values below will be raised to this value).</param>
    /// <param name="maxFiniteVal">Maximum allowed finite value (values above will be lowered to this value).</param>
    /// <param name="dumpStats">If true, outputs the count of elements truncated high and low.</param>
    /// <returns>The clipped tensor.</returns>
    public static Tensor ClippedSubnormalOrLarge(Tensor input, double minPositiveVal = 1e-7, double maxFiniteVal = 1e4, bool dumpStats = false)
    {
      Tensor result = input.clone();
      Tensor positiveClipped = torch.clamp(result, min: minPositiveVal, max: maxFiniteVal);
      Tensor negativeClipped = torch.clamp(result, min: -maxFiniteVal, max: -minPositiveVal);
      Tensor clipped = torch.where(result.gt(0), positiveClipped, torch.where(result.lt(0), negativeClipped, result));

      if (dumpStats)
      {
        Tensor positiveHighMask = result.gt(maxFiniteVal);
        Tensor positiveLowMask = result.gt(0).bitwise_and(result.lt(minPositiveVal));
        Tensor negativeLowMask = result.lt(-maxFiniteVal);
        Tensor negativeHighMask = result.lt(0).bitwise_and(result.gt(-minPositiveVal));

        long positiveHighCount = positiveHighMask.sum().item<long>();
        long positiveLowCount = positiveLowMask.sum().item<long>();
        long negativeLowCount = negativeLowMask.sum().item<long>();
        long negativeHighCount = negativeHighMask.sum().item<long>();

        long truncatedHigh = positiveHighCount + negativeHighCount;
        long truncatedLow = positiveLowCount + negativeLowCount;
        if (truncatedHigh + truncatedLow > 0)
        {
          Console.Write("min: " + input.min().item<float>() + ", max: " + input.max().item<float>() + "  ");
          Console.WriteLine("truncated High: " + truncatedHigh + ", truncated Low: " + truncatedLow);
        }
      }

      return clipped;
    }




    #region Copying parameters

    /// <summary>
    /// Copies all parameters from sourceNet into destNet, assuming they have the same structure.
    /// Returns (paramCount, layerCount) as a tuple.
    /// </summary>
    public static (long paramCount, long layerCount) CopyParameters(Module sourceNet, Module destNet)
    {
      if (sourceNet == null) throw new ArgumentNullException(nameof(sourceNet));
      if (destNet == null) throw new ArgumentNullException(nameof(destNet));

      long totalParamCount = 0;
      long totalLayerCount = 0;

      RecurseCopy(sourceNet, destNet, ref totalParamCount, ref totalLayerCount);

      // Optionally print or log the final result:
      Console.WriteLine($"[CopyParameters] Copied {totalParamCount} parameters across {totalLayerCount} layers.");

      return (totalParamCount, totalLayerCount);
    }

    /// <summary>
    /// Recursively copies parameters from source to destination, accumulating counts.
    /// </summary>
    static void RecurseCopy(Module source, Module dest, ref long paramCount, ref long layerCount)
    {
//      var nps = source.named_parameters().ToArray();

      object sourceParamsObj = source.GetType()
                                   .GetField("_internal_params", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                                   ?.GetValue(source);
      OrderedDict<string, Parameter> sourceParamsDict = sourceParamsObj as OrderedDict<string, Parameter>;
      object destParamsObj = dest.GetType()
                                   .GetField("_internal_params", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                                   ?.GetValue(source);
      OrderedDict<string, Parameter> destParamsDict = destParamsObj as OrderedDict<string, Parameter>;

      if (sourceParamsDict.Keys.Count != destParamsDict.Keys.Count)
      {
        throw new InvalidOperationException(
           $"Parameter count mismatch in modules '{source.GetType()}' vs '{dest.GetType()}'. " +
           $"Source has {sourceParamsDict.Keys.Count}, Dest has {destParamsDict.Keys.Count}." );
      }

      // Copy parameters at the current module level
      foreach ((string, Parameter) destParam in destParamsDict)
      {
        // Copies tensor data in-place
        destParam.Item2.copy_(sourceParamsDict[destParam.Item1]);
        paramCount++;
      }

      // Consider this a 'layer'
      layerCount++;

      // Recursively process child modules
      Module[] sourceChildren = source.children().ToArray();
      Module[] destChildren = dest.children().ToArray();

      if (sourceChildren.Length != destChildren.Length)
      {
        throw new InvalidOperationException(
            $"Submodule count mismatch in modules '{source.GetType()}' vs '{dest.GetType()}'. " +
            $"Source has {sourceChildren.Length}, Dest has {destChildren.Length}."
        );
      }

      for (int i = 0; i < sourceChildren.Length; i++)
      {
        RecurseCopy(sourceChildren[i], destChildren[i], ref paramCount, ref layerCount);
      }
    }

    #endregion
  }

}
