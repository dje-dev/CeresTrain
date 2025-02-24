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
using System.Linq;
using System.Collections.Generic;

using static TorchSharp.torch;
using TorchSharp.Modules;
using CeresTrain.Networks.MiscModules;
using static TorchSharp.torch.nn;
using Ceres.Base.Math;

#endregion

namespace CeresTrain.Utils
{
  /// <summary>
  /// Set of utilities for loading weights from a dictionary of weights into a module.
  /// </summary>
  public static class ModuleParamLoadingUtils
  {
    /// <summary>
    /// Loads weights from a dictionary of weights into a Linear layer.
    /// </summary>
    /// <param name="paramsSource"></param>
    /// <param name="paramsLoaded"></param>
    /// <param name="parameter"></param>
    /// <param name="weightsName"></param>
    /// <param name="biasesName"></param>
    public static void ParameterLoad(Dictionary<string, Tensor> paramsSource, HashSet<string> paramsLoaded, Parameter parameter, string weightsName)
    {
      Tensor weightsNew = GetParams(paramsSource, paramsLoaded, weightsName).to(parameter.device).to(parameter.dtype);
      CheckSizesSame(weightsName, parameter, weightsNew);
      parameter.set_(weightsNew);
    }


    /// <summary>
    /// Loads weights from a dictionary of weights into a Linear layer.
    /// </summary>
    /// <param name="paramsSource"></param>
    /// <param name="paramsLoaded"></param>
    /// <param name="linear"></param>
    /// <param name="weightsName"></param>
    /// <param name="biasesName"></param>
    public static void LinearLoad(Dictionary<string, Tensor> paramsSource, HashSet<string> paramsLoaded, Module<Tensor,Tensor> linearModule, string weightsName, string biasesName)
    {
      Linear linear = linearModule as Linear;
      Tensor weightsNew = GetParams(paramsSource, paramsLoaded, weightsName).to(linear.weight.dtype, linear.weight.device);

#if SHOW_WEIGHTS
      float[] weightsFlat = weightsNew.to(ScalarType.Float32).cpu().data<float>().ToArray();
      float min = (float)StatUtils.Min(weightsFlat);
      float max = (float)StatUtils.Max(weightsFlat);
      Console.WriteLine($"layer {weightsName}  min: {min,6:F2}  max: {max,6:F2}");
#endif

      CheckSizesSame(weightsName, linear.weight, weightsNew);
      linear.weight.copy_(weightsNew);

      if (biasesName is not null)
      {
        Tensor biasesNew = GetParams(paramsSource, paramsLoaded, biasesName).to(linear.bias.dtype, linear.bias.device);
        CheckSizesSame(weightsName, linear.bias, biasesNew);
        linear.bias.copy_(biasesNew);
      }
    }


    static Tensor GetParams(Dictionary<string, Tensor> paramsSource, HashSet<string> paramsLoaded, string name)
    {
      if (!paramsSource.ContainsKey(name))
      {
        throw new Exception("Parameters not found for name " + name);
      }

      Tensor weightsNew = paramsSource[name];
      if (weightsNew.IsInvalid)
      {
        throw new Exception("Parameters found in model but invalid, possibly disposed? " + name);
      }

      MarkParamUsed(name, paramsLoaded);
      return weightsNew;
    }


    /// <summary>
    /// Loads weights from a dictionary of weights into a LayerNorm layer.
    /// </summary>
    /// <param name="paramsSource"></param>
    /// <param name="paramsLoaded"></param>
    /// <param name="layerNorm"></param>
    /// <param name="weightsName"></param>
    /// <param name="biasesName"></param>
    public static void LayerNormLoad(Dictionary<string, Tensor> paramsSource, HashSet<string> paramsLoaded, LayerNorm layerNorm, string weightsName, string biasesName)
    {
      Tensor weightsNew = GetParams(paramsSource, paramsLoaded, weightsName).to(layerNorm.weight.dtype, layerNorm.weight.device);
      CheckSizesSame(weightsName, layerNorm.weight, weightsNew);
      layerNorm.weight.copy_(weightsNew);

      if (biasesName is not null)
      {
        Tensor biasesNew = GetParams(paramsSource, paramsLoaded, biasesName).to(layerNorm.bias.dtype, layerNorm.bias.device);

        CheckSizesSame(weightsName, layerNorm.bias, biasesNew);
        layerNorm.bias.copy_(biasesNew);
      }
    }


    /// <summary>
    /// Loads weights from a dictionary of weights into a RMSNormLoad layer.
    /// </summary>
    /// <param name="paramsSource"></param>
    /// <param name="paramsLoaded"></param>
    /// <param name="rmsNorm"></param>
    /// <param name="weightsName"></param>
    /// <param name="biasesName"></param>
    public static void RMSNormLoad(Dictionary<string, Tensor> paramsSource, HashSet<string> paramsLoaded, RMSNorm rmsNorm, string weightsName)
    {
      Tensor weightsNew = GetParams(paramsSource, paramsLoaded, weightsName).to(rmsNorm.Scale.dtype, rmsNorm.Scale.device);
      CheckSizesSame(weightsName, rmsNorm.Scale, weightsNew);
      rmsNorm.Scale.copy_(weightsNew);
    }


    static bool SizesSame(Parameter p1, Tensor p2) => p2.size().SequenceEqual(p1.size());
    static bool CheckSizesSame(string paramName, Parameter p1, Tensor p2)
      => SizesSame(p1, p2) ? true :
                             throw new ArgumentException(paramName + " parameter sizes not the same, "
                                                       + "network expects: " + TorchSharpUtils.ShapeStr(p1.size()) + " "
                                                       + "provided params: " + TorchSharpUtils.ShapeStr(p2.size()));

    static void MarkParamUsed(string paramName, HashSet<string> weightsLoaded)
    {
      if (weightsLoaded.Contains(paramName))
      {
        throw new ArgumentException("Parameter " + paramName + " already loaded");
      }
      else
      {
        weightsLoaded.Add(paramName);
      }
    }
  }

}
