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

using static CeresTrain.Utils.ModuleParamLoadingUtils;

#endregion

namespace CeresTrain.Networks.SoftMoE
{
  /// <summary>
  /// A more efficient alternative to creating 'n' separate expert layers (likely
  /// from 'nn.Linear' modules).  Instead, we create a single set of batched weights
  /// and biases, and apply all 'experts' in parallel.
  ///
  /// Transliterated from: https://github.com/fkodom/soft-mixture-of-experts/blob/main/soft_mixture_of_experts/multi_expert.py.
  /// </summary>
  public class LayerMultiExpert : Module<Tensor, Tensor>
  {
    /// <summary>
    /// Input dimension.
    /// </summary>
    public readonly int DimIn;

    /// <summary>
    /// Output dimension.
    /// </summary>
    public readonly int DimOut;

    /// <summary>
    /// Number of experts.
    /// </summary>
    public readonly int NumExperts;

    /// <summary>
    /// If a bias term should be included.
    /// </summary>
    public readonly bool UseBias;

    Parameter weight;
    Parameter bias_param;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="dimIn">embedding dimension</param>
    /// <param name="dimOut"></param>
    /// <param name="numExperts"></param>
    /// <param name="useBias"></param>
    /// <exception cref="NotImplementedException"></exception>
    public LayerMultiExpert(int dimIn, int dimOut, int numExperts, bool useBias = true) : base(nameof(LayerMultiExpert))
    {
      DimIn = dimIn;
      DimOut = dimOut;
      NumExperts = numExperts;
      UseBias = useBias;

      weight = Parameter(empty(numExperts, dimIn, dimOut));

      if (useBias)
      {
        bias_param = Parameter(empty(numExperts, dimOut));
      }

      reset_parameters();

      RegisterComponents();
    }


    void reset_parameters()
    {
      // NOTE: Mostly copy-pasted from 'nn.Linear.reset_parameters'
      // Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
      // uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
      // https://github.com/pytorch/pytorch/issues/57109

      init.kaiming_normal_(weight, a: Math.Sqrt(5));
      if (UseBias)
      {
        (long fan_in, long fan_out) = init.CalculateFanInAndFanOut(weight);
        float bound = fan_in > 0 ? 1.0f / MathF.Sqrt(fan_in) : 0.0f;
        init.uniform_(bias_param, -bound, bound);
      }
    }


    public void LoadWeights(Dictionary<string, Tensor> weightsSource, HashSet<string> weightsLoaded, int layerNum)
    {
      ParameterLoad(weightsSource, weightsLoaded, weight, $"transformer_layer.{layerNum}.moe.experts2.weight");
      ParameterLoad(weightsSource, weightsLoaded, bias_param, $"transformer_layer.{layerNum}.moe.experts2.bias");
    }


    /// <summary>
    /// Forward pass.
    /// 
    /// NOTE: Could probably be simplified, the following probably works (at least with num_slots==1):
    ///       However the efficiency (including memory use might be different/worse). 
    ///         x = torch.matmul(x, weight.unsqueeze(0));
    ///         return UseBias ? (x + bias_param.unsqueeze(0).unsqueeze(2)) : x;
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public override Tensor forward(Tensor x)
    {
      if (x.shape[^1] != DimIn)
      {
        throw new ArgumentException($"Expected input with embed_dim={DimIn} (dim=-1), but found {x.shape[^1]}");
      }

      if (x.shape[1] != NumExperts)
      {
        throw new ArgumentException($"Expected input with num_experts={NumExperts} (dim=1), but found {x.shape[1]}");
      }

      // NOTE: 'd1' and 'd2' are both equal to 'embed_dim'. But for 'einsum' to
      // work correctly, we have to give them different names.
      // FAILS!! x = einsum("b n ... d1, n d1 d2 -> b n ... d2", x, weight);
      // Instead replace d1 --> y, d2 --> z
      x = einsum("b n ... y, n y z -> b n ... z", x, weight);

      if (!UseBias)
      {
        return x;
      }
      else
      {
        Tensor bias_term;

        // NOTE: When used with 'SoftMoE' the inputs to 'MultiExpertLayer' will
        // always be 4-dimensional.  But it's easy enough to generalize for 3D
        // inputs as well, so I decided to include that here.
        if (x.shape.Length == 3)
        {
          // bias = rearrange(self.bias, "n d -> () n d")
          bias_term = bias_param.unsqueeze(0);
        }
        else if (x.shape.Length == 4)
        {
          //bias = rearrange(self.bias, "n d -> () n () d")
          bias_term = bias_param.unsqueeze(0).unsqueeze(2);
        }
        else
        {
          throw new NotImplementedException($"Expected input to have 3 or 4 dimensions, but got {x.shape.Length}");
        }

        return x + bias_term;
      }
    }
  }
}
