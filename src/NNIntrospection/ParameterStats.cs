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

using Ceres.Base.Math;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

#endregion

namespace CeresTrain.NNIntrospection
{
  /// <summary>
  /// Set of statistics relating to the various layers in a neural network.
  /// </summary>
  public class ParameterStats : List<ParameterStat>
  {
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="module"></param>
    public ParameterStats(Module module) : base(ExtractedParameterStats<Module, Module>(module))
    {
    }

    /// <summary>
    /// Dumps all statistics to Console.
    /// </summary>
    public void DumpAllStats() => DumpAllStats(Console.Out);


    /// <summary>
    /// Dumps all statistics to specified TextWriter.
    /// </summary>
    /// <param name="writer"></param>
    public void DumpAllStats(TextWriter writer)
    {
      // Dump all
      foreach (ParameterStat stat in this)
      {
        stat.Dump(writer);
      }
    }


    /// <summary>
    /// Returns a list of statistics for each parameter in the specified module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="TResult"></typeparam>
    /// <param name="module"></param>
    /// <returns></returns>
    static List<ParameterStat> ExtractedParameterStats<T, TResult>(Module module)
    {
      List<ParameterStat> stats = new List<ParameterStat>();
      foreach ((string name, Parameter parameter) param in module.named_parameters())
      {
        float[] paramValues = param.parameter.cpu().to(TorchSharp.torch.ScalarType.Float32).data<float>().ToArray();
        // TODO: collect all these stats in parallel with one pass over the data, using vectorized operations.
        ParameterStat stat = new ParameterStat(param.name,
                                             paramValues.Min(),
                                             paramValues.Max(),
                                             paramValues.Average(),
                                             (float)StatUtils.NormL2(paramValues),
                                             (float)StatUtils.StdDev(paramValues),
                                             (float)StatUtils.Kurtosis(paramValues));
        stats.Add(stat);
      }
      return stats;
    }

  }

}
