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

#endregion

#region Using directives

using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

#endregion

namespace CeresTrain.Networks.MiscModules
{
  /// <summary>
  /// Huber loss function with scaling factor.
  /// </summary>
  /// <typeparam name="T1"></typeparam>
  /// <typeparam name="T2"></typeparam>
  /// <typeparam name="TResult"></typeparam>
  public class HuberLossScaled<T1, T2, TResult> : Loss<Tensor, Tensor, Tensor>
  {
    /// <summary>
    /// Reduction type.
    /// </summary>
    public readonly Reduction Reduction;

    /// <summary>
    /// Delta base (starting point for outlier mitigation).
    /// </summary>
    public readonly float DeltaBase;

    /// <summary>
    /// Scaling factor to be applied after calculation.
    /// </summary>
    public readonly float PostCalcScale;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="reduction"></param>
    /// <param name="deltaBase"></param>
    /// <param name="postCalcScale"></param>
    public HuberLossScaled(Reduction reduction, float deltaBase, float postCalcScale) : base(reduction)
    {
      Reduction = reduction;
      DeltaBase = deltaBase;
      PostCalcScale = postCalcScale;
    }


    public override Tensor forward(Tensor output, Tensor target)
    {
      return PostCalcScale * functional.huber_loss(output, target, reduction: Reduction, delta: DeltaBase);
    }
  }

}
