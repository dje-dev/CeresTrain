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

namespace CeresTrain.Networks.SoftMoE
{
  /// <summary>
  /// Layer implementing L2 norm possibly with additional learnable scaling factor.
  /// </summary>
  class LayerL2NormScaled : Module<Tensor, Tensor>
  {
    const float EPSILON = 1E-6f;

    /// <summary>
    /// Index of the dimension along which to compute the norm.
    /// </summary>
    public readonly long IndexDimToNormalize;

    /// <summary>
    /// If true, the output is multiplied by a learnable scaling factor.
    /// </summary>
    public readonly bool WithScaleFactor;

    /// <summary>
    /// The optional learnable scaling factor.
    /// </summary>
    public Parameter ScaleFactorParam { get; }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="indexDimToNormalize"></param>
    /// <param name="withScaleFactor"></param>
    public LayerL2NormScaled(long indexDimToNormalize, bool withScaleFactor) : base(nameof(LayerL2NormScaled))
    {
      WithScaleFactor = withScaleFactor;
      IndexDimToNormalize = indexDimToNormalize;

      if (withScaleFactor)
      {
        // Scalar scaling factor (initial value 1).
        ScaleFactorParam = Parameter(ones(1));
      }

      RegisterComponents();
    }


    /// <summary>
    /// Forward function.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public override Tensor forward(Tensor x)
    {
      Tensor sumsSquared = x.pow(2).mean([IndexDimToNormalize], true) + EPSILON;
      Tensor normed = x * rsqrt(sumsSquared);
      return WithScaleFactor ? normed * ScaleFactorParam : normed;
    }
  }
}
