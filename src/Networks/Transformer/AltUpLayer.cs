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

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

#endregion

namespace CeresTrain.Networks.Transformer
{
  /// <summary>
  /// Alternating updates algorithm for transformer.
  /// See: https://arxiv.org/abs/2301.13310.
  /// </summary>
  public class AltUpLayer : Module<Tensor, Tensor, Tensor>
  {
    public readonly int K;
    public readonly int j_star;

    private Parameter P;
    private Parameter G;

    /// <summary>
    /// Constructor with specified number of side-by-side blocks 
    /// and which block which receives tranformer update.
    /// </summary>
    /// <param name="K"></param>
    /// <param name="j_star"></param>
    public AltUpLayer(int K, int j_star) : base("AltUp")
    {
      this.K = K;
      this.j_star = j_star;

      P = Parameter(torch.randn(new long[] { K, K }), requires_grad: true);
      G = Parameter(torch.randn(new long[] { K }), requires_grad: true);

      RegisterComponents();
    }


    public override Tensor forward(Tensor x_old, Tensor x_tilde_j_star)
    {
      // TODO: Training speed is very slow because of piecemeal tensor operations in loops below.
      // Rewrite the loops below to be loopless using tensor operations.

      // Predict.
      Tensor x_tilde = torch.zeros_like(x_old);
      for (int i = 0; i < K; i++)
      {
        using (NewDisposeScope())
        {
          Tensor newVal = torch.zeros_like(x_tilde_j_star);
          for (int j = 0; j < K; j++)
          {
            newVal += P[i, j] * x_old[TensorIndex.Colon, TensorIndex.Colon, j, TensorIndex.Colon];
          }
          x_tilde.index_put_(newVal, TensorIndex.Colon, TensorIndex.Colon, i, TensorIndex.Colon);
        }
      }

      // Correct.
      Tensor x_new = torch.zeros_like(x_old);
      Tensor unscaledCorrection = x_tilde_j_star - x_tilde[TensorIndex.Colon, TensorIndex.Colon, j_star, TensorIndex.Colon];
      for (int i = 0; i < K; i++)
      {
        using (NewDisposeScope())
        {
          Tensor correction = G[i] * unscaledCorrection;
          Tensor corrected = x_tilde[TensorIndex.Colon, TensorIndex.Colon, i, TensorIndex.Colon] + correction;
          x_new.index_put_(corrected, TensorIndex.Colon, TensorIndex.Colon, i, TensorIndex.Colon);
        }
      }

      return x_new;
    }
  }
  }
