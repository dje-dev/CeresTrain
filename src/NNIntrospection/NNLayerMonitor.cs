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

using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;

using Ceres.Base.Math;
using CeresTrain.Networks.Transformer;
using CeresTrain.Networks;

#endregion

namespace CeresTrain.NNIntrospection
{
  /// <summary>
  /// Monitors a specified neural network layer and tracks statistics related to the output of that layer.
  /// </summary>
  public class NNLayerMonitor
  {
    /// <summary>
    /// The parent monitor to which this layer monitor belongs.
    /// </summary>
    public readonly NNLayerMonitorSet ParentSet;

    /// <summary>
    /// Name of the layer being monitored.
    /// </summary>
    public readonly string LayerName;

    /// <summary>
    /// Number of times evaluation is skipped between evaluations.
    /// 
    /// Value of 0 turns of monitoring.
    /// 
    /// Values greater than 1 are typically used to sample a subset
    /// and thereby reduce the amount of monitoring overhead.
    /// </summary>
    public readonly int MonitorSkipCount;

    /// <summary>
    /// Number of times evaluation is skipped between evaluations.
    /// 
    /// Value of 0 turns of output of statistics.
    /// 
    /// Values greater than 1 are typically used to sample a subset
    /// and thereby reduce the verbosity of output.
    /// </summary>
    public readonly int DumpSkipCount;

    /// <summary>
    /// Optional supplementary statistic to track for this layer.
    /// </summary>
    public readonly SupplementaryStatType SupplementaryStat;

    /// <summary>
    /// Average of the standard deviation of the output tensor. 
    /// </summary>
    public float AvgSD => sumSD / count;

    /// <summary>
    /// Average of the kurtosis of the output tensor. 
    /// </summary>
    public float AvgKurtosis => sumKurtosis / count;

    /// <summary>
    /// Minimum of the mimima of the output tensor.
    /// </summary>
    public float MinMin { private set; get; } = float.MaxValue;

    /// <summary>
    /// Maximum of the maxima of the output tensor.
    /// </summary>
    public float MaxMax { private set; get; } = float.MinValue;

    /// <summary>
    /// Average of the minima of the output tensor.
    /// </summary>
    public float Average => sumAvg / count;

    /// <summary>
    /// Average of the supplementary statistic for this layer. 
    /// </summary>
    public float SupplementalStatAvg => supplementalStatAgg / count;


    #region Private state fields

    float sumAvg;
    float count;
    float sumSD;
    float sumKurtosis;

    float lastKurtosis;
    float lastSD;
    float lastMin;
    float lastMax;
    float lastAvg;

    int lastDumpSkipCount;
    int lastMonitorSkipCount;

    int lastBatchSize;
    Module ParentModule;
    Module LayerModule;

    internal Func<Module, Tensor, Tensor, float> supplementalStatCalcCallback;
    float supplementalStatAgg;

    #endregion


    /// <summary>
    /// Type of supplementary statistic associated with neural network layer.
    /// </summary>
    public enum SupplementaryStatType
    {
      None,

      /// <summary>
      /// Average cosine similarity between rightmost dimension of the output tensor.
      /// For example, if applied to the output of the attention layer (after the V projection)
      /// in a transformer this gives a measure of the degree of similarity (redundancy)
      /// between the attention heads.
      /// See for example section 3 of "Multi-Head Attention with Disagreement Regularization" by Li et. al.
      /// </summary>
      AverageCosineSimilarity,

      /// <summary>
      /// Measures the intrinsic dimensionality of the output tensor using the two nearest neighbors.
      /// See "Intrinsic dimension of data representations in deep neural networks" by Ansuini et. al.
      /// </summary>
      IntrinsicDimensionalityTwoNN
    }


    /// <summary>
    /// Creates a new NNLayerMonitor for the specified layer.
    /// </summary>
    /// <param name="module"></param>
    /// <param name="layerName"></param>
    /// <param name="monitorSkipCount">monitor every this many calls, or 0 for every time</param>
    /// <param name="dumpSkipCount">monitor every this many calls, or 0 for every time</param>
    /// <param name="supplementaryStat">type of additional statistic to monitor</param>
    /// <param name="parentSet"></param>
    public NNLayerMonitor(Module parentModule, Module layerModule, string layerName,
                          int monitorSkipCount, int dumpSkipCount = 0,
                          SupplementaryStatType supplementaryStat = SupplementaryStatType.None,
                          NNLayerMonitorSet parentSet = null)
    {
      ParentModule = parentModule;
      LayerModule = layerModule;
      LayerName = layerName;

      MonitorSkipCount = monitorSkipCount;
      DumpSkipCount = dumpSkipCount;
      SupplementaryStat = supplementaryStat;
      ParentSet = parentSet;

      if (supplementaryStat != SupplementaryStatType.None)
      {
        InstallSupplementaryStatCalc(supplementaryStat);
      }

      RegisterHook();
    }


    private void InstallSupplementaryStatCalc(SupplementaryStatType supplementaryStat)
    {
      supplementalStatCalcCallback = (module, input, output) =>
      {
        if (supplementaryStat == SupplementaryStatType.AverageCosineSimilarity)
        {
          // TODO: Remove this hardcoded class-specific logic.
          if (module is not NetTransformerLayerEncoder)
          {
            throw new Exception("Currently only CeresTransformerLayerEncoder is supported for AverageCosineSimilarity");
          }
          NetTransformerLayerEncoder encoderLayer = (NetTransformerLayerEncoder)module;
          float[] cosineSimilarities = encoderLayer.lastAttentionHeadOutputCosineSimilarities;
          if (cosineSimilarities == null)
          {
            throw new Exception("cosineSimilarities was null");
          }
          return StatUtils.Average(cosineSimilarities);
        }
        else if (supplementaryStat == SupplementaryStatType.IntrinsicDimensionalityTwoNN)
        {
          float[] intrinsicDimensionality = IntrinsicDimensionality.TwoNN(output);
          return intrinsicDimensionality[0];
        }
        else
        {
          throw new Exception("Unknown supplementary stat type: " + supplementaryStat);
        }
      };
    }


    /// <summary>
    /// Dumps the statistics for the last execution of the layer to the console.
    /// </summary>
    public void DumpStatsLast()
    {
      Console.WriteLine($"{LayerName,-50}  avg:{lastAvg,8:F3}  +/-{lastSD,8:F3}   kurt:{lastKurtosis,7:F1}    max:{lastMax,8:F3}  {SupplementalStatAvg,8:F3}  (batch size {lastBatchSize})");
    }


    /// <summary>
    /// Dumps the statistics for the most extreme execution of the layer to the console.
    /// </summary>
    public void DumpStatsMostExtreme()
    {
      string supplemental = supplementalStatCalcCallback == null ? "" : $"  {SupplementaryStat}: {SupplementalStatAvg,8:F3}";
      Console.WriteLine($"{LayerName,-50}  Avg:{Average,7:F3}    Avg SD:{AvgSD,7:F1}    Avg Kurt:{AvgKurtosis,7:F1}     Min Min:{MinMin,8:F3}     Max Max:{MaxMax,8:F3}   {supplemental}");
    }


    static (float sd, float kurt) AvgStatsByRightmost(Tensor orgTensor)
    {
      Tensor mean = torch.mean(orgTensor, new long[] { -1 });

      Tensor std = torch.std(orgTensor, -1);
      Tensor zero_centered = (orgTensor - mean.unsqueeze(-1)).pow(4);
      Tensor kurtosis = torch.mean(zero_centered, new long[] { -1 }) / std.pow(4);
      return (std.mean().cpu().ToSingle(), kurtosis.mean().cpu().ToSingle());
    }


    /// <summary>
    /// Registers a hook for the specified layer.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="layerName"></param>
    void RegisterHook()
    {
      Console.WriteLine("Registering hook for layer: " + LayerName);
      RegisterHook(ParentModule, LayerName, (layer, input, output) =>
      {
        ParentSet?.ChildCallback(this);

        if (MonitorSkipCount == 0 || ++lastMonitorSkipCount % MonitorSkipCount != 0)
        {
          return output;
        }

        lastBatchSize = (int)output.shape[0];

        using (NewDisposeScope())
        {
          //          Console.WriteLine($"Hook executed for layer: {layerName} {output.shape}");
          // TODO: Retrive as Half to reduce bytes transferred over the bus
          //float[] outputF = new TensorAccessor<float>(output.to(ScalarType.Float32).cpu()).ToArray();

          lastMax = output.max().cpu().ToSingle();
          lastMin = output.min().cpu().ToSingle();
          lastAvg = output.mean().cpu().ToSingle();

          // The statistics have to be calculated only over the
          // the rightmost dimension and averaged.
          (float lastSD, float lastKurtosis) = AvgStatsByRightmost(output);

          sumAvg += lastAvg;
          count++;

          sumSD += lastSD;
          sumKurtosis += lastKurtosis;

          MinMin = Math.Min(MinMin, lastMin);
          MaxMax = Math.Max(MaxMax, lastMax);

          //  Calculate any supplemental statistics.
          if (supplementalStatCalcCallback != null)
          {
            supplementalStatAgg += supplementalStatCalcCallback(LayerModule, input, output);
          }

          if (DumpSkipCount > 0 && ++lastDumpSkipCount % DumpSkipCount != 0)
          {
            DumpStatsLast();
          }

          return output;
        }
      });
    }


    public void SetSupplementalStat(SupplementaryStatType stat)
    {
      InstallSupplementaryStatCalc(stat);
    }


    public void SetSupplementalStat(Func<Module, Tensor, Tensor, float> statCalcCallback)
    {
      supplementalStatCalcCallback = statCalcCallback;
    }


    void RegisterHook(Module model, string layerName, Func<Module<Tensor, Tensor>, Tensor, Tensor, Tensor> callback)
    {
      foreach ((string name, Module<Tensor, Tensor> module) node in model.named_modules())
      {
        if (layerName == node.name)
        {
          // Set up (optional) pre-hook to inform the layer if monitoring will be called for the next invocation. 
          node.module.register_forward_pre_hook((module, input) =>
          {
            IModuleReceivesMonitoringStatusInfo interfacePreMonitor = node.module as IModuleReceivesMonitoringStatusInfo;
            if (interfacePreMonitor != null)
            {
              bool monitoringActive = MonitorSkipCount != 0 || (lastMonitorSkipCount + 1) % MonitorSkipCount == 0;

              interfacePreMonitor.MonitoringCurrentInvocation = monitoringActive;
            }
            return input;
          });

          // Set up the hook to monitor the layer.  
          node.module.register_forward_hook((module, input, output) =>
          {
            return callback(node.module, input, output);
          });

          return;
        }
      }

      throw new Exception($"Could not find layer {layerName} in model");
    }

  }
}
