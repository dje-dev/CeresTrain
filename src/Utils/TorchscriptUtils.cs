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
using System.IO;
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
    public static ScriptModule<T, TResult> TorchScriptFilesAveraged<T, TResult>(string tsFileName1, string tsFileName2, 
                                                                                Device device, ScalarType dataType)
    {
      if (tsFileName2 == null)
      {
        return load<T, TResult>(tsFileName1, device.type, device.index).to(dataType);
      }
      else
      {
        return TorchScriptFilesAveraged<T, TResult>(new string[] { tsFileName1, tsFileName2 }, null, device, dataType);
      }
    }


    /// <summary>
    /// Returns a TorchScript module created from a weighted average of the TorchScript files.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="tsFileName1"></param>
    /// <param name="tsFileName2"></param>
    /// <param name="device"></param>
    /// <param name="dataType"></param>
    /// <returns></returns>
    public static ScriptModule<T, TResult> TorchScriptFilesAveraged<T, TResult>(
        string[] tsFileNames, float[] tsWeights, Device device, ScalarType dataType)
    {
      if (tsFileNames == null || tsFileNames.Length == 0)
      {
        throw new ArgumentException("No file names provided.", nameof(tsFileNames));
      }

      if (tsWeights != null && tsWeights.Length != tsFileNames.Length)
      {
        throw new ArgumentException("Weights array must be the same length as file names array.", nameof(tsWeights));
      }

      if (tsFileNames.Any(fn => string.IsNullOrEmpty(fn)))
      {
        throw new ArgumentException("One or more provided file names are null or empty.", nameof(tsFileNames));
      }

      // Load each file as a module
      ScriptModule<T, TResult>[] modules =  new ScriptModule<T, TResult>[tsFileNames.Length];
      for (int i = 0; i < tsFileNames.Length; i++)
      {
        if (!File.Exists(tsFileNames[i]))
        {
          throw new ArgumentException($"File {tsFileNames[i]} does not exist.", nameof(tsFileNames));
        }

        modules[i] = load<T, TResult>(tsFileNames[i], device.type, device.index).to(dataType);
      }

      // Apply weighted averaging to the modules
      return TorchScriptModulesAverage(modules, tsWeights);
    }


    /// <summary>
    /// Returns weighted average of TorchScript modules.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="modules"></param>
    /// <param name="weights"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static ScriptModule<T, TResult> TorchScriptModulesAverage<T, TResult>(ScriptModule<T, TResult>[] modules, float[] weights = null)
    {
      if (modules == null || modules.Length == 0)
      {
        throw new ArgumentException("No modules provided.", nameof(modules));
      }

      if (weights != null && weights.Length != modules.Length)
      {
        throw new ArgumentException("Weights array must be the same length as modules array.", nameof(weights));
      }

      if (modules.Any(m => m == null))
      {
        throw new ArgumentException("One or more provided modules are null.", nameof(modules));
      }

      if (weights == null)
      {
        // Set weights to be array of equal values summing to 1.0.
        weights = new float[modules.Length];
        Array.Fill(weights, 1.0f / weights.Length); 
      }

      // Initialize dictionary to hold weighted sum of parameters.
      Dictionary<string, Tensor> weightedSumDict = new ();

      // Compute weighted sum of parameters from all modules.
      for (int i = 0; i < modules.Length; i++)
      {
        var module = modules[i];
        float weight = weights[i];

        foreach ((string name, Parameter parameter) in module.named_parameters())
        {
          parameter.requires_grad = false;
          Tensor weightedParam = parameter.mul(weight);

          if (weightedSumDict.ContainsKey(name))
          {
            weightedSumDict[name] = weightedSumDict[name].add(weightedParam);
          }
          else
          {
            weightedSumDict[name] = weightedParam; // No need to clone as mul returns a new tensor
          }
        }
      }

      // Update the first module's parameters with the weighted average.
      foreach ((string name, Parameter parameter) in modules[0].named_parameters())
      {
        Tensor weightedAverageTensor = weightedSumDict[name];
        weightedAverageTensor.requires_grad = false;
        parameter.set_(new Parameter(weightedAverageTensor, false));
      }

      modules[0].eval();
      return modules[0];
    }


  }

}
