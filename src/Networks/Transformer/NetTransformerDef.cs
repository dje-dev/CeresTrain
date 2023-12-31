﻿#region License notice

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
      SwiGLU
    }


    [Flags]
    public enum TransformerFeatures
    {
      // No extra features.
      None = 0,

      /// <summary>
      /// Turn on Smolgen feature with typical sizing.
      /// </summary>
      Smolgen = 1,

      /// <summary>
      /// Turn on Soft Mixture of Experts feature with typical hyperparameters.
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

      if (extraFeatures.HasFlag(TransformerFeatures.Attention2x))
      {
        AttentionMultiplier = 2;
      }

      if (extraFeatures.HasFlag(TransformerFeatures.Smolgen))
      {
        SmolgenDimPerSquare = 8;
        SmolgenDim = 64;
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
    /// If transformer encoded block should be pre-normalized (as opposed to post-normalized).
    /// </summary>
    public readonly bool PreNorm { get; init; } = false;

    /// <summary>
    /// Type of normalization to be applied within Encoder blocks.
    /// </summary>
    public readonly NormalizationType NormType { get; init; } = NormalizationType.LayerNorm;

    /// <summary>
    /// Multiplier for the attention heads (dimensionality upscaled by this factor before being split into heads).
    /// </summary>
    public readonly int AttentionMultiplier { get; init; } = 1;

    /// <summary>
    /// Factor by which the FFN inner hidden layer is larger than the model dimension.
    /// </summary>
    public readonly int FFNMultiplier { get; init; } = 1;

    /// <summary>
    /// Type of activation function used between layers of the FFN.
    /// </summary>
    public readonly ActivationType FFNActivationType { get; init; } = ActivationType.ReLUSquared;

    /// <summary>
    /// Activation function to use in network heads.
    /// </summary>
    public readonly ActivationType HeadsActivationType { get; init; } = ActivationType.ReLU;

    /// <summary>
    /// If true, use deep normalization (with scaling of residual connection).
    /// NOTE: the deepnorm implementation may be incomplete (weight initialization possibly missing).
    /// </summary>
    public readonly bool DeepNorm { get; init; } = false;

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

      if (NumLayers < 1 || ModelDim < 1 || FFNMultiplier < 1 || AttentionMultiplier < 1 || HeadWidthMultiplier < 1)
      {
        throw new Exception($"One of NumLayers/ModelDim/FFNMultiplier/AttentionMultiplier/HeadWidthMultiplier is too small (less than 1).");
      }
    }

    #endregion
  }
}
