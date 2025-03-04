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

    public Parameter Scale;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="dim"></param>
    /// <param name="eps"></param>
    public RMSNorm(int dim, float eps = 1e-6f) : base(nameof(RMSNorm))
    {
      Epsilon = eps;
      Scale = Parameter(ones(dim));

      RegisterComponents();
    }


    /// <summary>
    /// Forward pass.
    /// 
    /// Note that some networks will show numerical instability 
    /// with a naive 16 bit implementation, so we upcast to avoid this.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public override Tensor forward(Tensor x)
    {
      // Upcast x to float32 for more robust accumulation.
      Tensor x32 = x.to(ScalarType.Float32);

      // Compute the root mean square along the last dimension.
      Tensor rms = torch.sqrt(torch.mean(x32.pow(2), dimensions: [-1], keepdim: true));

      // Normalize the tensor.
      Tensor xNormalized = x32 / (rms + Epsilon);

      // Cast back to the original type and multiply by the scale factor.
      Tensor result = (xNormalized.to(x.dtype) * Scale);

      return result;
    }
  }
}
