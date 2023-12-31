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

using static TorchSharp.torch.nn;
using static TorchSharp.torch;

#endregion

namespace CeresTrain.NNIntrospection
{
  /// <summary>
  /// Monitors the set of all neural network layers and tracks statistics related to the output of those layers.  
  /// </summary>
  public class NNLayerMonitorSet
  {
    /// <summary>
    /// Set of all layers being monitored.
    /// </summary>
    public List<NNLayerMonitor> Monitors { get; private set; }

    /// <summary>
    /// Number of times evaluation is skipped between evaluations.
    /// 
    /// Value of 0 turns of output of statistics.
    /// 
    /// Values greater than 1 are typically used to sample a subset
    /// and thereby reduce the verbosity of output.
    /// </summary>
    public readonly int DumpSkipCount;


    int lastDumpSkipCount;


    /// <summary>
    /// Constructs a new NNLayerMonitorSet for the specified module.
    /// </summary>
    /// <param name="module"></param>
    /// <param name="monitorSkipCount">monitor every this many calls, or 0 for every time</param>
    /// <param name="dumpSkipCount">dump every this many calls, or 0 for every time</param>
    /// <param name="layerIncludeFilter">predicate returning if the layer should be included in those monitored, or null for all</param>
    /// <param name="layerSupplementaryStat">predicate returning what supplementary stat should be used with a given layer, or null for None</param>
    public NNLayerMonitorSet(Module module,
                             int monitorSkipCount = 0,
                             int dumpSkipCount = 0,
                             Predicate<string> layerIncludeFilter = null,
                             Func<string, NNLayerMonitor.SupplementaryStatType> layerSupplementaryStat = null)
    {
      DumpSkipCount = dumpSkipCount;
      Monitors = new List<NNLayerMonitor>();

      foreach ((string name, Module module) node in module.named_modules())
      {
        if (layerIncludeFilter == null || layerIncludeFilter(node.name))
        {
          // Possibly also register a supplementary stat for this layer.
          NNLayerMonitor.SupplementaryStatType stat = NNLayerMonitor.SupplementaryStatType.None;
          if (layerSupplementaryStat != null)
          {
            stat = layerSupplementaryStat(node.name);
          }

          Monitors.Add(new NNLayerMonitor(module, node.module, node.name, monitorSkipCount, 0, stat, this));
        }
      }
    }

    /// <summary>
    /// Sets the supplementary stat for the specified layer.
    /// </summary>
    /// <param name="layerName"></param>
    /// <param name="stat"></param>
    /// <exception cref="ArgumentException"></exception>
    public void SetSupplementalLayerStat(string layerName, NNLayerMonitor.SupplementaryStatType stat)
    {
      int indexLayer = Monitors.FindIndex(monitor => monitor.LayerName == layerName);
      if (indexLayer == -1)
      {
        throw new ArgumentException($"Layer name {layerName} not found in monitor set.");
      }

      Monitors[indexLayer].SetSupplementalStat(stat);
    }


    /// <summary>
    /// Sets the supplementary stat for the specified layer to a specified value.
    /// </summary>
    /// <param name="layerName"></param>
    /// <param name="computeLayerStatFunc"></param>
    /// <exception cref="ArgumentException"></exception>
    public void SetSupplementalLayerStat(string layerName, Func<Module, Tensor, Tensor, float> computeLayerStatFunc)
    {
      int indexLayer = Monitors.FindIndex(monitor => monitor.LayerName == layerName);
      if (indexLayer == -1)
      {
        throw new ArgumentException($"Layer name {layerName} not found in monitor set.");
      }

      Monitors[indexLayer].SetSupplementalStat(computeLayerStatFunc);
    }


    /// <summary>
    /// Method called by every child monitor upon each call into them.
    /// </summary>
    /// <param name="childMonitor"></param>
    internal void ChildCallback(NNLayerMonitor childMonitor)
    {
      // Only consider possibly dumping stats once per network
      // (arbitrarily using the last layer as the trigger).
      if (ReferenceEquals(childMonitor, Monitors[^1]))
      {
        if (DumpSkipCount == 0 || ++lastDumpSkipCount % DumpSkipCount == 0)
        {
          Console.WriteLine();
          DumpStatsMostExtreme();
        }
      }
    }


    /// <summary>
    /// Dumps the last execution statistics for all layers to the console.
    /// </summary>
    public void DumpStatsLast()
    {
      foreach (NNLayerMonitor monitor in Monitors)
      {
        monitor.DumpStatsLast();
      }
    }


    /// <summary>
    /// Dumps the execution statistics (most extreme observations) for all layers to the console.
    /// </summary>
    public void DumpStatsMostExtreme()
    {
      foreach (NNLayerMonitor monitor in Monitors)
      {
        monitor.DumpStatsMostExtreme();
      }
    }
  }
}
