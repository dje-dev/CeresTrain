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
using System.Text.Json.Serialization;

#endregion

namespace CeresTrain.Networks.SoftMoE
{
  /// <summary>
  /// Parameters which define how soft Mixture of Experts (MoE) should be used.
  /// 
  /// See: "From Sparse to Soft Mixtures of Experts" (https://arxiv.org/pdf/2308.00951.pdf)
  /// </summary>
  public readonly record struct SoftMoEParams
  {
    [JsonIgnore]
    public bool SecondLayerOnly => MoEMode == SoftMoEModeType.ReplaceLinearSecondLayer
                                || MoEMode == SoftMoEModeType.AddLinearSecondLayer;

    public enum SoftMoEModeType
    {
      /// <summary>
      /// Replace the FFN layer completely with a SoftMoE (comprised of two linear layers).
      /// </summary>
      ReplaceLinear,

      /// <summary>
      /// Compute SoftMoE (comprised of two linear layers) and add it to the FFN layer.
      /// </summary>
      AddLinear,

      /// <summary>
      /// Replace the second linear layer of the FFN layer with a SoftMoE (single layer).
      /// </summary>
      ReplaceLinearSecondLayer,

      /// <summary>
      /// Add the output of a SoftMoE (single layer) to the output of the second linear layer of the FFN layer.
      /// </summary>
      AddLinearSecondLayer
    };


    /// <summary>
    /// Type of SoftMoE.
    /// </summary>
    public readonly SoftMoEModeType MoEMode { get; init; } = SoftMoEModeType.AddLinearSecondLayer;


    /// <summary>
    /// If true, SoftMoE only used for alternating layers (where layer index mod 2 is 1).
    /// </summary>
    public bool OnlyForAlternatingLayers { get; init; } = true;

    /// <summary>
    /// Number of experts to use (or 0 for no SoftMoE). 
    /// </summary>
    public readonly int NumExperts { get; init; } = 0;

    /// <summary>
    /// Number of slots to allocate per expert.
    /// </summary>
    public readonly int NumSlotsPerExpert { get; init; } = 1;

    /// <summary>
    /// If normalization should be applied to x and Phi before computing dispatch logits.
    /// May only be necessary with pre-norm nets, or with very large nets.
    /// </summary>
    public readonly bool UseNormalization { get; init; } = false;

    /// <summary>
    /// If bias should be used by the expert linear layers.
    /// </summary>
    public readonly bool UseBias { get; init; } = true;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="moeMode"></param>
    /// <param name="numExperts"></param>
    /// <param name="numSlotsPerExpert"></param>
    /// <param name="onlyForAlternatingLayers"></param>
    public SoftMoEParams(SoftMoEModeType moeMode, int numExperts, int numSlotsPerExpert, bool onlyForAlternatingLayers)
    {
      MoEMode = moeMode;
      NumExperts = numExperts;
      NumSlotsPerExpert = numSlotsPerExpert;
      OnlyForAlternatingLayers = onlyForAlternatingLayers;
    }
  }
}
