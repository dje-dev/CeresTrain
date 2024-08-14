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

using CeresTrain.Networks.SoftMoE;
using CeresTrain.Trainer;

#endregion

namespace CeresTrain.Networks.Transformer
{

  /// <summary>
  /// Definition of shape/features of a transformer neural network.
  /// </summary>
  public readonly record struct NetTransformerDef : ICeresNeuralNetDef
  {
    public enum NormalizationType
    {
      /// <summary>
      /// No normalization.
      /// </summary>
      None,

      /// <summary>
      /// Layer normalization.
      /// </summary>
      LayerNorm,

      /// <summary>
      /// RMS normalization (see "Root Mean Square Layer Normalization" by Zhang, Sennrich).
      /// </summary>
      RMSNorm
    }


    public enum ActivationType
    {
      /// <summary>
      /// No activation (identify function).
      /// </summary>
      None,

      /// <summary>
      /// Rectified linear unit.
      /// </summary>
      ReLU,

      /// <summary>
      /// Swish activation.
      /// </summary>
      Swish,

      /// <summary>
      /// Rectified linear unit with squared output.
      /// </summary>
      ReLUSquared,

      /// <summary>
      /// SwiGLU activation ("GLU Variants Improve Transformer" by Noam Shazeer)
      /// </summary>
      SwiGLU,

      /// <summary>
      /// Mish activation ("Mish: A Self Regularized Non-Monotonic Activation Function" by Diganta Misra)
      /// </summary>
      Mish
    }

    /// <summary>
    /// Type of secondary attention mode (if any).
    /// </summary>
    public enum DualAttentionModeType
    {
      /// <summary>
      /// No secondary attention.
      /// </summary>
      None,

      /// <summary>
      /// Dual attention only.
      /// </summary>
      DualAttentionOnly,

      /// <summary>
      /// Dual attention and feedforward network.
      /// Based on "DaViT: Dual Attention Vision Transformers" by Ding et. al.
      /// </summary>
      DualAttentionAndFFN,
    }


    [Flags]
    public enum TransformerFeatures
    {
      // No extra features.
      None = 0,

      /// <summary>
      /// Turn on Smolgen feature with typical sizing.
      /// See: https://lczero.org/blog/2024/02/transformer-progress/ (Daniel Moore).
      /// </summary>
      Smolgen = 1,

      /// <summary>
      /// Turn on Soft Mixture of Experts feature with typical hyperparameters.
      /// See "From Sparse to Soft Mixtures of Experts" by Puigcerver et. al.
      /// </summary>
      SoftMoE = 2,

      /// <summary>
      /// Turn on attention replication feature (2x) whereby
      /// the attention takes place over an expended dimension of size 2x the embedding dimension.
      /// </summary>
      Attention2x = 4 // Represents Attention2x
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="modelDim"></param>
    /// <param name="numLayers"></param>
    /// <param name="numHeads"></param>
    /// <param name="ffnMultiplier"></param>
    /// <param name="extraFeatures"></param>
    public NetTransformerDef(int modelDim, int numLayers, int numHeads, int ffnMultiplier, TransformerFeatures extraFeatures)
    {
      ModelDim = modelDim;
      NumLayers = numLayers;
      NumHeads = numHeads;
      FFNMultiplier = ffnMultiplier;
      NormType = NormalizationType.RMSNorm;

      if (extraFeatures.HasFlag(TransformerFeatures.Attention2x))
      {
        AttentionMultiplier = 2;
      }

      if (extraFeatures.HasFlag(TransformerFeatures.Smolgen))
      {
        SmolgenDimPerSquare = 32;
        SmolgenDim = 256; // tried 512 here, with head divisor 2, but many more parameters and at most 12 Elo better
        SmolgenActivationType = ActivationType.None;
        SmolgenToHeadDivisor = 1;
      }

      if (extraFeatures.HasFlag(TransformerFeatures.SoftMoE))
      {
        SoftMoEConfig = new SoftMoEParams() with
        {
          NumExperts = 16,
          MoEMode = SoftMoEParams.SoftMoEModeType.AddLinearSecondLayer,
          NumSlotsPerExpert = 1,
          OnlyForAlternatingLayers = true,
          UseBias = true,
          UseNormalization = false
        };
      }
    
    }


    /// <summary>
    /// Null constructor for deserialization.
    /// </summary>
    [JsonConstructor]
    public NetTransformerDef()
    {
    }


    /// <summary>
    /// Number of hidden dimensions in the model.
    /// </summary>
    public readonly int ModelDim { get; init; } = 256;

    /// <summary>
    /// Number of layers in the model.
    /// </summary>
    public readonly int NumLayers { get; init; } = 8;

    /// <summary>
    /// Number of attention heads in the model.
    /// </summary>
    public readonly int NumHeads { get; init; } = 8;

    /// <summary>
    /// Type of dual attention (if any).
    /// </summary>
    public readonly DualAttentionModeType DualAttentionMode { get; init; } = DualAttentionModeType.None;

    /// <summary>
    /// If transformer encoded block should be pre-normalized (as opposed to post-normalized).
    /// Contrary to modern practice with language models, PostNorm is found to have significantly lower loss.
    /// (perhaps because the model is not as deep as in language models and convergence problems do not arise).
    /// </summary>
    public readonly bool PreNorm { get; init; } = false;

    /// <summary>
    /// Type of normalization to be applied within Encoder blocks.
    /// RMSNorm is found slightly faster but same accuracy as LayerNorm.
    /// </summary>
    public readonly NormalizationType NormType { get; init; } = NormalizationType.RMSNorm;

    /// <summary>
    /// Multiplier for the attention heads (dimensionality upscaled by this factor before being split into heads).
    /// </summary>
    public readonly int AttentionMultiplier { get; init; } = 1;

    /// <summary>
    /// Factor by which the FFN inner hidden layer is larger than the model dimension.
    /// </summary>
    public readonly int FFNMultiplier { get; init; } = 4;

    /// <summary>
    /// Type of activation function used between layers of the FFN.
    /// </summary>
    public readonly ActivationType FFNActivationType { get; init; } = ActivationType.Mish;

    /// <summary>
    /// Activation function to use in network heads.
    /// </summary>
    public readonly ActivationType HeadsActivationType { get; init; } = ActivationType.Mish;

    /// <summary>
    /// Dimension of the vector (per square) passed between consecutive positions.
    /// </summary>
    public readonly int PriorStateDim { get; init; } = 0;

    /// <summary>
    /// If the KQV matrices used in attention should computed with some nonlinearity/MLP.
    /// This idea was applied by Muhan Zhang to language models in 2023, see:
    /// "Neural Attention: Enhancing QKV Calculation in Self-Attention Mechanism with Neural Networks"
    /// https://arxiv.org/pdf/2310.11398.
    /// </summary>
    public readonly bool NonLinearAttention { get; init; } = false;

    /// <summary>
    /// If true, use deep normalization (with scaling of residual connection).
    /// NOTE: the deepnorm implementation may be incomplete (weight initialization possibly missing).
    /// See: "DeepNet: Scaling Transformers to 1,000 Layers" (2022) by Wang et. al. (https://arxiv.org/abs/2203.00555).
    /// </summary>
    public readonly bool DeepNorm { get; init; } = false;

    /// <summary>
    /// If true, the DenseFormer architecture is used. See:
    ///   "DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging," Pagliardini et. al.  
    /// </summary>
    public readonly bool DenseFormer { get; init; } = false;

    /// <summary>
    /// Number of per square dimensions used for Smolgen (or 0 if Smolgen not used).
    /// Invented by Ergodice, see: https://github.com/Ergodice/lczero-training.
    /// </summary>
    public readonly int SmolgenDimPerSquare { get; init; } = 0;

    /// <summary>
    /// Number of dimensions in intermediate layer for Smolgen (or 0 if Smolgen not used).
    /// Invented by Ergodice, see: https://github.com/Ergodice/lczero-training.
    /// </summary>
    public readonly int SmolgenDim { get; init; } = 0;

    /// <summary>
    /// Divisor applied SmolgenDim for to per-head sizing of prep layers in Smolgen (mapping to 64x64 attention).
    /// </summary>
    public readonly int SmolgenToHeadDivisor { get; init; } = 1;

    /// <summary>
    /// Type of activation to use for Smolgen layers.
    /// Lc0 nets my have used swish, but simple linear (no activation) seemingly also found good.
    /// </summary>
    public readonly ActivationType SmolgenActivationType { get; init; } = ActivationType.None;

    /// <summary>
    /// If relative positional encoding should be used (for Q, K and V).
    /// </summary>
    public readonly bool UseRPE { get; init; } = true;

    /// <summary>
    /// If relative bias should be used for RPE.
    /// </summary>
    public readonly bool UseRelBias { get; init; } = false;

    /// <summary>
    /// Multiplier applied to the width of the default size of each layers in the output heads.
    /// </summary>
    public readonly int HeadWidthMultiplier { get; init; } = 4;


    /// <summary>
    /// The soft mixture of networks (MoE) configuration.
    /// </summary>
    public readonly SoftMoEParams SoftMoEConfig { get; init; } = new SoftMoEParams();

    #region Helper methods

    /// <summary>
    /// Returns a short description of the main network characteristics.
    /// </summary>
    public override string ToString()=> $"Transformer ({ModelDim}x{NumLayers}x{NumHeads} FFN {FFNMultiplier})";


    /// <summary>
    /// Factory method to create an actual neural network from the definition.
    /// </summary>
    /// <param name="netConfig"></param>
    /// <returns></returns>
    public CeresNeuralNet CreateNetwork(in ConfigNetExecution netConfig) => new NetTransformer(netConfig, this);


    /// <summary>
    /// Check if the configuration is valid.
    /// </summary>
    public void Validate()
    {
      if (ModelDim % NumHeads != 0)
      {
        throw new Exception($"ModelDim ({ModelDim}) must be divisible by NumHeads ({NumHeads})");
      }

      if (NumLayers < 1 || ModelDim < 1 || AttentionMultiplier < 1 || HeadWidthMultiplier < 1)
      {
        throw new Exception($"One of NumLayers/ModelDim/AttentionMultiplier/HeadWidthMultiplier is too small (less than 1).");
      }
    }

    #endregion

    /// <summary>
    /// Reserved value used for debugging/experimentation to turn on a possible ad hoc test/diagnostic feature.
    /// </summary>
    public readonly float TestValue { get; init; } = 0;

  }
}
