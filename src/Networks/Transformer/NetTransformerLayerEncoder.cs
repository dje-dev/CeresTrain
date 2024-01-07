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
using System.Diagnostics;
using System.Collections.Generic;

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

using CeresTrain.NNIntrospection;
using CeresTrain.Networks.SoftMoE;
using CeresTrain.Networks.MiscModules;
using static CeresTrain.Utils.ModuleParamLoadingUtils;
using CeresTrain.Utils;

#endregion

namespace CeresTrain.Networks.Transformer
{
  /// <summary>
  /// Encoder layer used in CeresTransformerEncoder.
  /// </summary>
  internal class NetTransformerLayerEncoder : Module<Tensor, Tensor>, IModuleReceivesMonitoringStatusInfo
  {
    /// <summary>
    /// Epsilon used for normalization. 
    /// </summary>
    const float NORM_EPS = 1e-6f;

    /// <summary>
    /// Index of this layer within the stack.
    /// </summary>
    public readonly int LayerNum;

    /// <summary>
    /// Number of layers in the encoder stack.
    /// </summary>
    public readonly int NumLayers;
    
    /// <summary>
    /// Model embedding dimension.
    /// </summary>
    public readonly int DimModel;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    public readonly int NumHeads;

    /// <summary>
    /// Type of normalization to use.
    /// </summary>
    public readonly NetTransformerDef.NormalizationType NormType;

    /// <summary>
    /// Multiplier for FFN expansion factor.
    /// </summary>
    public readonly int FFNMult;

    /// <summary>
    /// Activation function to use in FFN.
    /// </summary>
    readonly NetTransformerDef.ActivationType FFNActivation;

    /// <summary>
    /// If true, use pre-normalization (as opposed to post-normalization).
    /// </summary>
    public readonly bool PreNorm;

    /// <summary>
    /// Number of neurons to be used per square if Smolgen is enabled (otherwise 0).
    /// </summary>
    public readonly int SmolgenPerSquareDim;

    /// <summary>
    /// Size of intermediate layer in Smolgen.
    /// </summary>
    public readonly int SmolgenIntermediateDim;

    /// <summary>
    /// Fraction of units to drop out.
    /// </summary>
    public readonly float DropoutRate;

    /// <summary>
    /// Multiplier to be used when possibly expanding the attention head dimensionality.
    /// </summary>
    public readonly int AttentionMultiplier;

    /// <summary>
    /// If dropout should be applied during inference.
    /// </summary>
    public readonly bool DropoutDuringInference;

    /// <summary>
    /// If true, monitor the cosine similarity of the attention head output with the output weights.
    /// </summary>
    public readonly bool MonitorCosineSimilarity;

    /// <summary>
    /// Alpha scaling coefficient (for use with DeepNorm).
    /// </summary>
    public readonly float Alpha;

    /// <summary>
    /// Dimension (width) of each attention head.
    /// </summary>
    int DimPerHead => DimModel / NumHeads;


    internal float[] lastAttentionHeadOutputCosineSimilarities;
    public bool MonitoringCurrentInvocation
    {
      get;
      set;
    }

    public readonly float ClipLevel;
    public readonly bool MonitorMoEActivationStats;

    Linear attentionQKV;

    Linear attentionOutput;

    Linear mlpLinear1;
    Linear mlpLinear2;
    Linear mlpLinear3;

    Module<Tensor, Tensor> norm1;
    Module<Tensor, Tensor> norm2;
    ReLU relu;

    Dropout dropoutAttention;
    Dropout dropoutMLP;

    Linear smolgenLinear1;
    Linear smolgenLinear2;
    Module<Tensor, Tensor> smLN1;
    Linear smolgenLinear3;
    Module<Tensor, Tensor> smLN2;
    WrappedLinear smolgenLinearShared;
    LayerSoftMoEBatchedDual moe;

    SoftMoEParams SoftMoEParams;



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="numLayers"></param>
    /// <param name="layerNum"></param>
    /// <param name="dim"></param>
    /// <param name="numHeads"></param>
    /// <param name="preNorm"></param>
    /// <param name="attentionMultiplier"></param>
    /// <param name="ffnMult"></param>
    /// <param name="alpha"></param>
    /// <param name="dropoutRate"></param>
    /// <param name="dropoutDuringInference"></param>
    /// <param name="clipLevel"></param>
    /// <param name="monitorCosineSimilarity"></param>
    /// <param name="smolgenPerSquareDim"></param>
    /// <param name="smolgenIntermediateDim"></param>
    /// <param name="softMoEParams"></param>
    /// <param name="monitorMoEActivationStats"></param>
    /// <param name="smLinearShared"></param>
    /// <exception cref="ArgumentException"></exception>
    public NetTransformerLayerEncoder(int numLayers, int layerNum, int dim, int numHeads, bool preNorm,
                                      NetTransformerDef.NormalizationType normType,
                                      int attentionMultiplier, int ffnMult, NetTransformerDef.ActivationType ffnActivation,
                                      float alpha, float dropoutRate, bool dropoutDuringInference, float clipLevel,
                                      bool monitorCosineSimilarity,
                                      int smolgenPerSquareDim, int smolgenIntermediateDim,
                                      SoftMoEParams softMoEParams, bool monitorMoEActivationStats, ref Linear smLinearShared)
      : base($"transformer_layer.{layerNum}")
    {
      if (dim % numHeads != 0)
      {
        throw new ArgumentException($"dim ({dim}) must be divisible by numHeads ({numHeads})");
      }

      NumLayers = numLayers;
      LayerNum = layerNum;
      DimModel = dim;
      NumHeads = numHeads;
      Alpha = alpha;
      FFNMult = ffnMult;
      NormType = normType;
      FFNActivation = ffnActivation;
      PreNorm = preNorm;
      AttentionMultiplier = attentionMultiplier;
      DropoutRate = dropoutRate;
      DropoutDuringInference = dropoutDuringInference;
      ClipLevel = clipLevel;
      MonitorCosineSimilarity = monitorCosineSimilarity;
      SoftMoEParams = softMoEParams;
      SmolgenPerSquareDim = smolgenPerSquareDim;
      SmolgenIntermediateDim = smolgenIntermediateDim;
      MonitorMoEActivationStats = monitorMoEActivationStats;

      attentionQKV = Linear(dim, dim * attentionMultiplier * 3, hasBias: false);
      attentionOutput = Linear(dim * attentionMultiplier, dim, hasBias: true);

      bool useSoftMoE = (softMoEParams.MoEMode != SoftMoEParams.SoftMoEModeType.None 
                     && softMoEParams.NumExperts > 0)
                     && (!softMoEParams.OnlyForAlternatingLayers || (layerNum % 2 == 1));

      if (useSoftMoE)
      {
        moe = new LayerSoftMoEBatchedDual(dim, dim * ffnMult,
                                          num_experts: softMoEParams.NumExperts,
                                          slots_per_expert: softMoEParams.NumSlotsPerExpert,
                                          useNormalization: softMoEParams.UseNormalization,
                                          onlySecondLayer: softMoEParams.SecondLayerOnly,
                                          bias: softMoEParams.UseBias,
                                          monitorMoEActivationStats: monitorMoEActivationStats,
                                          layerNum: layerNum);
      }

      const bool USE_FFN_BIAS = false;

      if (!useSoftMoE || softMoEParams.MoEMode != SoftMoEParams.SoftMoEModeType.ReplaceLinear)
      {
        mlpLinear1 = Linear(dim, dim * ffnMult, hasBias: USE_FFN_BIAS);
      }

      if (!useSoftMoE || softMoEParams.MoEMode != SoftMoEParams.SoftMoEModeType.ReplaceLinear &&
                          softMoEParams.MoEMode != SoftMoEParams.SoftMoEModeType.ReplaceLinearSecondLayer)
      {
        mlpLinear2 = Linear(dim * ffnMult, dim, hasBias: USE_FFN_BIAS);
      }

      if (FFNActivation == NetTransformerDef.ActivationType.SwiGLU)
      {
        mlpLinear3 = Linear(dim, dim * ffnMult, hasBias: USE_FFN_BIAS);
      }

      norm1 = MakeNormalizationLayer(DimModel);
      norm2 = MakeNormalizationLayer(DimModel);

      if (dropoutRate > 0.0f)
      {
        dropoutAttention = Dropout(dropoutRate);
        dropoutMLP = Dropout(dropoutRate);
      }

      relu = ReLU(inplace: true);

      if (smolgenPerSquareDim > 0)
      {
        SmolgenAttentionInit(ref smLinearShared);
      }

      RegisterComponents();
    }

    /// <summary>
    /// Create a normalization layer of the appropriate type.
    /// </summary>
    /// <param name="layerDim"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    private Module<Tensor, Tensor> MakeNormalizationLayer(int layerDim)
    {
      return NormType switch
      {
        NetTransformerDef.NormalizationType.RMSNorm => new RMSNorm(layerDim, NORM_EPS),
        NetTransformerDef.NormalizationType.LayerNorm => LayerNorm(layerDim, NORM_EPS),
        NetTransformerDef.NormalizationType.None => Identity(),
        _ => throw new NotImplementedException("Unsupported NormalizationType: " + NormType)
      };
    } 


    /// <summary>
    /// Helper class to wrap a Linear layer.
    /// This has the effect of making it invisible to RegisterComponents
    /// and is necessary when we want to share a Linear layer (which is already registered elsewhere).
    /// </summary>
    /// <param name="linear"></param>
    class WrappedLinear(Linear linear) 
    {
      public Linear Linear { get; } = linear;
    }


    /// <summary>
    /// Initializes thelayers related to Smolgen.
    /// </summary>
    /// <param name="sm4Shared"></param>
    void SmolgenAttentionInit(ref Linear sm4Shared)
    {

      if (SmolgenPerSquareDim > 0)
      {
        smolgenLinear1 = Linear(DimModel, SmolgenPerSquareDim);
        smolgenLinear2 = Linear(64 * SmolgenPerSquareDim, SmolgenIntermediateDim);
        smLN1 = MakeNormalizationLayer(SmolgenIntermediateDim);
        smolgenLinear3 = Linear(SmolgenIntermediateDim, NumHeads * SmolgenIntermediateDim);
        smLN2 = MakeNormalizationLayer(NumHeads * SmolgenIntermediateDim);
        smolgenLinearShared = new(sm4Shared);
      }
    }


    /// <summary>
    /// Computes Smolgen attention logits.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    Tensor SmolgenAttentionCalc(Tensor x)
    {
      using (NewDisposeScope())
      {
        x = smolgenLinear1.call(x); // --> (64, 32) = (batch, S, SM_PER_SQUARE)
        x = x.reshape(-1, 64 * SmolgenPerSquareDim); // --> 2048 = (batch, 64 * SM_PER_SQUARE)
        x = smolgenLinear2.call(x); // --> 256 = (batch, SM_INTERMEDIATE)
        x = smLN1.call(x);
        x = smolgenLinear3.call(x); // --> 256 * H (batch, H * SM_INTERMEDIATE)
        x = smLN2.call(x);
        x = smolgenLinearShared.Linear.call(x); // --> 64 * 64 = (batch, H, S, S)
        x = x.reshape(-1, 64, 64);

#if DEBUG
        if (float.IsNaN(x.sum().to(ScalarType.Float32).cpu().item<float>()))
          throw new Exception("Null smolgen");
#endif
        return x.MoveToOuterDisposeScope();
      }
    }


    /// <summary>
    /// Computes the scaled dot product attention.
    /// </summary>
    /// <param name="Q"></param>
    /// <param name="K"></param>
    /// <param name="V"></param>
    /// <param name="clipGamma"></param>
    /// <param name="smolgenLogits"></param>
    /// <returns></returns>
    Tensor ScaledDotProductAttention(Tensor Q, Tensor K, Tensor V, float clipGamma, Tensor smolgenLogits)
    {
      using (NewDisposeScope())
      {
        Tensor scores = matmul(Q, K.transpose(2, 3));
        scores.div_(MathF.Sqrt(DimPerHead));

        if (SmolgenPerSquareDim > 0)
        {
          Tensor smolgenLogitsRepeated = smolgenLogits.repeat(1, NumHeads, 1, 1).reshape(smolgenLogits.shape[0], NumHeads, 64, 64);
          scores += smolgenLogitsRepeated;
        }

        Tensor A = functional.softmax(scores, dim: -1);

        if (clipGamma != 0)
        {
          Debug.Assert(clipGamma < 0);
          float gamma = clipGamma;
          float zeta = 1.0f;

          A = A * (zeta - gamma) + gamma;
          A = clip(A, 0, 1);
        }

        Tensor H = matmul(A, V);
        return H.MoveToOuterDisposeScope();
      }
    }


    public float[] CalcAttentionHeadOutputCosineOutputSimilarities(Tensor attentionScoresOutput)
    {
      return VectorSimilarity.CalcAttentionHeadOutputCosineOutputSimilarities(attentionScoresOutput, attentionOutput.weight, DimModel, NumHeads, DimPerHead);
    }

    /// <summary>
    /// Forward propagation method.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    /// <exception cref="NotImplementedException"></exception>
    public override Tensor forward(Tensor x)
    {
      bool isEval = !training;
      using (NewDisposeScope())
      {
        int batch_size = (int)x.size(0);

        Tensor attentionInput = PreNorm ? norm1.call(x) : x;

        Tensor qkv = attentionQKV.call(attentionInput);

        // Split apart Q, K, V (with heads on the left)
        qkv = qkv.reshape(batch_size, 64, NumHeads, 3 * DimPerHead);
        qkv = qkv.permute(0, 2, 1, 3);
        Tensor[] qkvChunks = qkv.chunk(3, -1);
        (Tensor attentionQ, Tensor attentionK, Tensor attentionV) = (qkvChunks[0], qkvChunks[1], qkvChunks[2]);

        float thisClipLevel = ClipLevel != 0 && LayerNum == NumLayers - 1 ? ClipLevel : 0;
        bool monitorThisTime = MonitorCosineSimilarity && MonitoringCurrentInvocation;

        Tensor H_cat;
        if (SmolgenPerSquareDim == 0 && thisClipLevel == 0 && !monitorThisTime)
        {
          H_cat = functional.scaled_dot_product_attention(attentionQ, attentionK, attentionV);
        }
        else
        {
          Tensor smolgenLogits = null;
          if (SmolgenPerSquareDim > 0)
          {
            smolgenLogits = SmolgenAttentionCalc(x);
          }

          H_cat = ScaledDotProductAttention(attentionQ, attentionK, attentionV, thisClipLevel, smolgenLogits);


          if (monitorThisTime)
          {
            lastAttentionHeadOutputCosineSimilarities = VectorSimilarity.CalcAttentionHeadOutputCosineOutputSimilarities(H_cat, attentionOutput.weight, DimModel, NumHeads, DimPerHead);
          }
        }

        // Put all the heads back together by concat (with heads moved back to the right).
        Tensor H_cat1 = H_cat.transpose(1, 2).contiguous().view(batch_size, -1, DimModel * AttentionMultiplier);

        Tensor attn_output = attentionOutput.call(H_cat1);

        if (DropoutRate > 0)
        {
          if (isEval)
          {
            attn_output = functional.dropout(attn_output, DropoutRate, training: DropoutDuringInference);
          }
          else
          {
            attn_output = dropoutAttention.call(attn_output);
          }
        }

        // RESIDUAL/LAYER NORM
        Tensor out1 = x * Alpha + attn_output;
        if (!PreNorm)
        {
          out1 = norm1.call(out1);
        }


        // MLP
        Tensor mlpOutput;
        Tensor mlpInput = PreNorm ? norm2.call(out1) : out1;

        if (moe == null || SoftMoEParams.NumExperts == 0)
        {
          Tensor afterLinear1 = mlpLinear1.call(mlpInput);
          Tensor beforeLinear2 = TorchSharpUtils.WithActivation(afterLinear1, FFNActivation);
          if (FFNActivation == NetTransformerDef.ActivationType.SwiGLU)
          {
            beforeLinear2 *= mlpLinear3.call(afterLinear1);
          }
          mlpOutput = mlpLinear2.call(beforeLinear2);
        }
        else
        {
          if (FFNActivation != NetTransformerDef.ActivationType.ReLUSquared)
          {
            // TODO: make this work
            throw new Exception("Implementation limitation: only ReLUSquared supported with MoE");
          }

          switch (SoftMoEParams.MoEMode)
          {
            case SoftMoEParams.SoftMoEModeType.ReplaceLinear:
              throw new Exception("How to handle prenorm here?");
              mlpOutput = moe.call(out1);
              break;

            case SoftMoEParams.SoftMoEModeType.ReplaceLinearSecondLayer:
              Tensor mlp1b = relu.call(mlpLinear1.call(mlpInput)).square();
              mlpOutput = moe.call(mlp1b);
              break;

            case SoftMoEParams.SoftMoEModeType.AddLinear:
              Tensor mlp1 = relu.call(mlpLinear1.call(mlpInput)).square();
              mlpOutput = mlpLinear2.call(mlp1);
              mlpOutput += moe.call(out1);
              break;

            case SoftMoEParams.SoftMoEModeType.AddLinearSecondLayer:
              Tensor mlp1a = relu.call(mlpLinear1.call(mlpInput)).square();
              mlpOutput = mlpLinear2.call(mlp1a);
              mlpOutput += moe.call(mlp1a);
              break;

            default:
              throw new NotImplementedException();
          }
        }

        if (DropoutRate > 0)
        {
          if (isEval)
          {
            mlpOutput = functional.dropout(mlpOutput, DropoutRate, training: DropoutDuringInference);
          }
          else
          {
            mlpOutput = dropoutMLP.call(mlpOutput);
          }
        }

        // RESIDUAL/LAYER NORM
        Tensor out2 = out1 * Alpha + mlpOutput;

        if (!PreNorm)
        {
          out2 = norm2.call(out2);
        }

        return out2.MoveToOuterDisposeScope();
      }
    }

    #region Helper methods


    /// <summary>
    /// Loads weights from a dictionary.
    /// </summary>
    /// <param name="weightsSource"></param>
    /// <param name="weightsLoaded"></param>
    /// <exception cref="NotImplementedException"></exception>
    internal void LoadWeights(Dictionary<string, Tensor> weightsSource, HashSet<string> weightsLoaded)
    {
      LinearLoad(weightsSource, weightsLoaded, attentionQKV, $"transformer_layer.{LayerNum}.attention.qkv.weight", null);
      LinearLoad(weightsSource, weightsLoaded, attentionOutput, $"transformer_layer.{LayerNum}.attention.W_h.weight", $"transformer_layer.{LayerNum}.attention.W_h.bias");

      if (SoftMoEParams.NumExperts == 0 || SoftMoEParams.MoEMode != SoftMoEParams.SoftMoEModeType.ReplaceLinear)
      {
        LinearLoad(weightsSource, weightsLoaded, mlpLinear1, $"transformer_layer.{LayerNum}.mlp.linear1.weight", null);
      }

      if (FFNActivation == NetTransformerDef.ActivationType.SwiGLU)
      {
        LinearLoad(weightsSource, weightsLoaded, mlpLinear3, $"transformer_layer.{LayerNum}.mlp.linear3.weight", null);
      }

      if (SoftMoEParams.NumExperts == 0
      || (SoftMoEParams.MoEMode != SoftMoEParams.SoftMoEModeType.ReplaceLinearSecondLayer
       && SoftMoEParams.MoEMode != SoftMoEParams.SoftMoEModeType.ReplaceLinear))
      {
        LinearLoad(weightsSource, weightsLoaded, mlpLinear2, $"transformer_layer.{LayerNum}.mlp.linear2.weight", null);
      }

      if (NormType == NetTransformerDef.NormalizationType.LayerNorm)
      {
        LayerNormLoad(weightsSource, weightsLoaded, (LayerNorm)norm1, $"transformer_layer.{LayerNum}.ln1.weight", $"transformer_layer.{LayerNum}.ln1.bias");
        LayerNormLoad(weightsSource, weightsLoaded, (LayerNorm)norm2, $"transformer_layer.{LayerNum}.ln2.weight", $"transformer_layer.{LayerNum}.ln2.bias");
      }
      else if (NormType == NetTransformerDef.NormalizationType.RMSNorm)
      {
        RMSNormLoad(weightsSource, weightsLoaded, (RMSNorm)norm1, $"transformer_layer.{LayerNum}.ln1.scale");
        RMSNormLoad(weightsSource, weightsLoaded, (RMSNorm)norm2, $"transformer_layer.{LayerNum}.ln2.scale");
      }
      else if (NormType != NetTransformerDef.NormalizationType.None)
      {
        throw new NotImplementedException("Unsupported NormalizationType: " + NormType);
      }

      if (moe != null)
      {
        moe.LoadWeights(weightsSource, weightsLoaded, LayerNum);
      }

      if (SmolgenPerSquareDim > 0)
      {
        LinearLoad(weightsSource, weightsLoaded, smolgenLinear1, $"transformer_layer.{LayerNum}.attention.sm1.weight", $"transformer_layer.{LayerNum}.attention.sm1.bias");
        LinearLoad(weightsSource, weightsLoaded, smolgenLinear2, $"transformer_layer.{LayerNum}.attention.sm2.weight", $"transformer_layer.{LayerNum}.attention.sm2.bias");
        LinearLoad(weightsSource, weightsLoaded, smolgenLinear3, $"transformer_layer.{LayerNum}.attention.sm3.weight", $"transformer_layer.{LayerNum}.attention.sm3.bias");

        if (NormType == NetTransformerDef.NormalizationType.LayerNorm)
        {
          LayerNormLoad(weightsSource, weightsLoaded, (LayerNorm)smLN1, $"transformer_layer.{LayerNum}.attention.ln1.weight", $"transformer_layer.{LayerNum}.attention.ln1.bias");
          LayerNormLoad(weightsSource, weightsLoaded, (LayerNorm)smLN2, $"transformer_layer.{LayerNum}.attention.ln2.weight", $"transformer_layer.{LayerNum}.attention.ln2.bias");
        }
        else if (NormType == NetTransformerDef.NormalizationType.RMSNorm)
        {
          RMSNormLoad(weightsSource, weightsLoaded, (RMSNorm)smLN1, $"transformer_layer.{LayerNum}.attention.ln1.scale");
          RMSNormLoad(weightsSource, weightsLoaded, (RMSNorm)smLN2, $"transformer_layer.{LayerNum}.attention.ln2.scale");
        }
        else if (NormType != NetTransformerDef.NormalizationType.None)
        {
          throw new NotImplementedException("Unsupported NormalizationType: " + NormType);
        }

      }
    }


    #endregion
  }
}
