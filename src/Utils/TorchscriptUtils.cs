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
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.jit;

#endregion

namespace CeresTrain.Utils
{
  /// <summary>
  ///  Static helper methods relating to TorchScript API.
  /// </summary>
  public static class TorchscriptUtils
  {
    /// <summary>
    /// Returns a TorchScript module from a file, averaging in a second TorchScript module if provided.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="tsFileName1"></param>
    /// <param name="tsFileName2"></param>
    /// <param name="device"></param>
    /// <param name="dataType"></param>
    /// <returns></returns>
    public static ScriptModule<T, TResult> TorchScriptFilesAveraged<T, TResult>(string tsFileName1, string tsFileName2, Device device, ScalarType dataType)
    {
      ArgumentException.ThrowIfNullOrEmpty(tsFileName1, nameof(tsFileName1));

      ScriptModule<T, TResult> module1 = load<T, TResult>(tsFileName1, device.type, device.index).to(dataType);
      module1.eval();

      if (tsFileName2 == null)
      {
        return module1;
      }

      ScriptModule<T, TResult> module2 = load<T, TResult>(tsFileName2, device.type, device.index).to(dataType);
      return TorchScriptModulesAveraged(module1, module2);
    }


    /// <summary>
    /// Returns a TorchScript module averaged with another Torchscript module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="module1"></param>
    /// <param name="module2"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static ScriptModule<T, TResult> TorchScriptModulesAveraged<T, TResult>(ScriptModule<T, TResult> module1, ScriptModule<T, TResult> module2)
    {
      if (module2 == null)
      {
        throw new ArgumentException(nameof(module2));
      }

      // Build dictionary of parameters from module2.
      Dictionary<string, (string name, Parameter parameter)> p2Dict = module2.named_parameters().ToDictionary(s => s.name);

      // Average in parameters.
      foreach ((string name, Parameter parameter) param in module1.named_parameters())
      {
        Parameter p1 = param.parameter;
        p1.requires_grad = false;
        p2Dict[param.name].parameter.requires_grad = false;

        // Add in the second parameter and divide by 2 to get average.
        Tensor averageTensor = p1.add(p2Dict[param.name].parameter).div(2);
        averageTensor.requires_grad = false;
        param.parameter.set_(new Parameter(averageTensor, false));
      }
      return module1;
    }

  }

}
