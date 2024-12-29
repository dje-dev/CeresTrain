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

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

#endregion

namespace CeresTrain.Networks.MiscModules
{
  /// <summary>
  /// Root Mean Square Normalization.
  /// </summary>
  public class RMSNorm : Module<Tensor, Tensor>
  {
    public readonly float Epsilon;

    public Parameter Scale { get; internal set; }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="dim"></param>
    /// <param name="eps"></param>
    public RMSNorm(int dim, float eps = 1e-5f) : base(nameof(RMSNorm))
    {
      Epsilon = eps;
      Scale = Parameter(ones(dim));

      RegisterComponents();
    }

    Tensor Norm(Tensor x) => x * rsqrt(x.pow(2).mean([-1], true) + Epsilon);


    /// <summary>
    /// Forward pass.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public override Tensor forward(Tensor x)
    {
      Tensor rms = sqrt(mean(x.pow(2), dimensions: [-1], keepdim: true));

      Tensor x_normalized = x / (rms + Epsilon);

      return x_normalized * Scale;
    }
  }
}
