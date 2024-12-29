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
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


#endregion

namespace CeresTrain.Networks.MiscModules
{
  /// <summary>
  /// SwiGLU activation function.
  /// 
  /// Note: quick test on transformer showed large decrease in speed and no improvement in accuracy.
  /// </summary>
  public class SwiGLU : Module<Tensor, Tensor>
  {
    /// <summary>
    /// Dimension of the input.
    /// </summary>
    public readonly int Dim;

    /// <summary>
    /// If the layer has a bias term.
    /// </summary>
    public readonly bool HasBias;

    /// <summary>
    /// The linear layer.
    /// </summary>
    Linear dense;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="dim"></param>
    /// <param name="hasBias"></param>
    public SwiGLU(int dim, bool hasBias = true) : base(nameof(SwiGLU))
    {
      throw new NotImplementedException("Needs remediation to implement LoadWeights");

      HasBias = hasBias;
      Dim = dim;

      dense = Linear(dim, 2 * dim, hasBias: hasBias);
    }


    /// <summary>
    /// Forward pass.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public override Tensor forward(Tensor x)
    {
      x = dense.call(x);
      Tensor[] split = x.chunk(2, -1);
      (Tensor x1, Tensor x2) = (split[0], split[1]);
      return functional.SiLU(x1) * x2;
    }
  }
}
