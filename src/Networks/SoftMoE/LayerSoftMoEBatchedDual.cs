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
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using CeresTrain.Utils;

#endregion

namespace CeresTrain.Networks.SoftMoE
{
  /// <summary>
  /// A PyTorch module for Soft-MoE, as described in the paper:
  ///   "From Sparse to Soft Mixtures of Experts"
  ///    https://arxiv.org/pdf/2308.00951.pdf
  ///    
  /// This code mostly transliterated from: https://github.com/fkodom/soft-mixture-of-experts/tree/main
  /// 
  /// However:
  ///   - utilizes LayerMultiExpert for a "batched" implementation of the linear layers for efficiency
  ///   - flexibly supports replacing or supplementing either or both of the linear layers in the FFN
  ///   
  /// Note also:
  ///   - implements the full two-layer MLP with squared RELU nonlinear between the layers
  ///     (the same mixing is computed and used for both layers)
  ///   - normalization to x and Phi are added as an option (as described in section 2.3 of the paper)
  /// </summary>
  public class LayerSoftMoEBatchedDual : Module<Tensor, Tensor>
  {
    /// <summary>
    /// If activation statistics should be tracked.
    /// </summary>
    public readonly bool MonitorMoEActivationStats = false;
    public SoftMoEActivationInfo LastMoEInfo { get; private set; }

    public static SoftMoEActivationInfo[] LastMoEInfoStatic = new SoftMoEActivationInfo[30]; // TODO: Remove, temporary

    public readonly int LayerNum = -1;

    /// <summary>
    /// Width of input (embedding size).
    /// </summary>
    int Dim;

    /// <summary>
    /// Intermediate width of the FFN (between first and second linear layers).
    /// </summary>
    int FFNDim;

    /// <summary>
    /// Number of experts.
    /// </summary>
    int NumExperts;

    /// <summary>
    /// Number of slots per expert.
    /// </summary>
    int SlotsPerExpert;

    bool bias = true;

    /// <summary>
    /// If L2 normalization option should be enabled.
    /// </summary>
    bool useNormalization;

    /// <summary>
    /// If this module is used for the second layer only.
    /// </summary>
    bool onlySecondLayer;

    Parameter phi;

    LayerMultiExpert experts1;
    LayerMultiExpert experts2;

    LayerL2NormScaled normX;
    LayerL2NormScaled normPhi;

    /*
      Einstein notation:
    - b: batch size
    - m: input sequence length
    - d: embedding dimension
    - n: num experts
    - p: num slots per expert
    - (n * p): total number of slots

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        slots_per_expert (int): number of slots per expert (p)
        bias (bool): whether to include a bias term. Default: True.
   */
    public LayerSoftMoEBatchedDual(int dim, int ffnDim,
                                   int num_experts, int slots_per_expert,
                                   bool onlySecondLayer,
                                   bool useNormalization,
                                   bool bias = true,
                                   bool monitorMoEActivationStats = false,
                                   int layerNum = -1) : base("SoftMOEBatched")
    {
      Dim = dim;
      FFNDim = ffnDim;
      NumExperts = num_experts;
      SlotsPerExpert = slots_per_expert;
      this.useNormalization = useNormalization;
      this.onlySecondLayer = onlySecondLayer;
      this.bias = bias;
      MonitorMoEActivationStats = monitorMoEActivationStats;
      LayerNum = layerNum;

      int phiDim = onlySecondLayer ? ffnDim : dim;
      phi = Parameter(empty(phiDim, num_experts, slots_per_expert));

      if (!onlySecondLayer)
      {
        experts1 = new LayerMultiExpert(dim, ffnDim, num_experts, bias);
      }
      experts2 = new LayerMultiExpert(ffnDim, dim, num_experts, bias);

      if (useNormalization)
      {
        throw new Exception("normalization disabled until LoadWeights updated to accomodate");

        // See section 2.3 of the Soft MoE paper
        normX = new LayerL2NormScaled(1, false);
        normPhi = new LayerL2NormScaled(0, true);
      }

      reset_parameters();

      RegisterComponents();
    }


    void reset_parameters()
    {
      // NOTE: Copy weight initialization from 'nn.Linear.reset_parameters'
      // TODO: Check for initialization strategy from the paper
      // DISABLED: init.kaiming_uniform_(phi, a: MathF.Sqrt(5));
    }


    public void LoadWeights(Dictionary<string, Tensor> weightsSource, HashSet<string> weightsLoaded, int layerNum)
    {
      ModuleParamLoadingUtils.ParameterLoad(weightsSource, weightsLoaded, phi, $"transformer_layer.{layerNum}.moe.phi");

      if (experts1 != null)
      {
        experts1.LoadWeights(weightsSource, weightsLoaded, layerNum);
      }
      experts2.LoadWeights(weightsSource, weightsLoaded, layerNum);
    }


    /// <summary>
    /// Forward pass for the Soft-MoE layer, as described in:
    ///   https://arxiv.org/pdf/2308.00951.pdf
    /// See: equations(1-3), algorithm 1, and figure 2
    ///
    /// Einstein notation:
    ///        - b: batch size
    ///        - m: input sequence length
    ///        - d: embedding dimension
    ///        - n: num experts
    ///        - p: num slots per expert
    ///        - (n * p): total number of slots

    /// </summary>
    /// <param name="x">input tensor of shape(b, m, d)/param>
    /// <returns>output tensor of shape(b, m, d)</returns>
    public override Tensor forward(Tensor x)
    {
      int expectedDim = onlySecondLayer ? FFNDim : Dim;
      if (x.shape[^1] != expectedDim)
      {
        throw new ArgumentException($"Expected input with embed_dim={expectedDim} (dim=-1), but found {x.shape[^1]}");
      }

      if (x.shape.Length != 3)
      {
        throw new ArgumentException($"Expected input to have 3 dimensions, but got {x.shape.Length}.");
      }

      using (NewDisposeScope())
      {
        Tensor xPrepared;
        Tensor phiPrepared;

        if (useNormalization)
        {
          // Normalize inputs before computing logits
          xPrepared = normX.call(x);
          phiPrepared = normPhi.call(phi);
        }
        else
        {
          xPrepared = x;
          phiPrepared = phi;
        }

        // Github official code here:
        // https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py

        Tensor logits = einsum("b m d, d n p -> b m n p", xPrepared, phiPrepared);
        Tensor dispatch_weights = logits.softmax(dim: 1); // denoted 'D' in the paper.

        Tensor combine_weights;
        if (SlotsPerExpert == 1)
        {
          combine_weights = logits.softmax(dim: 2); // denoted 'C' in the paper.
        }
        else
        {
          // NOTE: The 'torch.softmax' function does not support multiple values for the
          // 'dim' argument (unlike JAX), so we are forced to flatten the last two dimensions.
          // Then, we rearrange the Tensor into its original shape.
          Tensor rearrangeTarget = logits.flatten(start_dim: 2).softmax(dim: -1); // denoted 'C' in the paper

          /// NOTE: The Torchsharp doesn't support rearrange function, so we use reshape instead.
          /// combine_weights = rearrange(rearrangeTarget, "b m (n p) -> b m n p", n: num_experts);
          long b = rearrangeTarget.shape[0];
          long m = rearrangeTarget.shape[1];
          long z = rearrangeTarget.shape[2];
          combine_weights = rearrangeTarget.reshape(b, m, NumExperts, z / NumExperts);
        }

        if (MonitorMoEActivationStats)
        {
          if (SlotsPerExpert != 1)
          {
            throw new NotImplementedException("slots_per_expert != 1");
          }

          LastMoEInfo = new(LayerNum,
                            dispatch_weights.transpose(1, 2).FloatArray3D(NumExperts, 64),
                            combine_weights.FloatArray3D(64, NumExperts));
          LastMoEInfoStatic[LayerNum] = LastMoEInfo;
        }

        x = einsum("b m d, b m n p -> b n p d", x, dispatch_weights); // Xs

        if (!onlySecondLayer)
        {
          Tensor linear1 = experts1.call(x); // First linear layer
          x = functional.relu(linear1).square(); // Squared RELU nonlinearity
        }
        Tensor linearOut = experts2.call(x); // Second linear layer
        Tensor combined = einsum("b n p d, b m n p -> b m d", linearOut, combine_weights); // Y

        return combined.MoveToOuterDisposeScope();
      }
    }
  }
}
