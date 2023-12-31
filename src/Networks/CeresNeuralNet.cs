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

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

using Ceres.Base.DataTypes;
using CeresTrain.NNEvaluators;

#endregion

namespace CeresTrain.Networks
{
  /// <summary>
  /// Base class from which all Ceres neural networks are derived.  
  /// </summary>
  public abstract class CeresNeuralNet : Module<Tensor, (Tensor, Tensor, Tensor, Tensor)>, IModuleNNEvaluator
  {
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="name"></param>
    public CeresNeuralNet(string name) : base(name)
    {

    }

    /// <summary>
    /// The forward method.
    /// </summary>
    /// <param name="inputSquares"></param>
    /// <param name="inputMoves"></param>
    /// <returns></returns>
    public abstract (Tensor value, Tensor policy, Tensor mlh, Tensor unc, FP16[] extraStats0, FP16[] extraStats1) Forward(Tensor inputSquares, Tensor inputMoves);


    public virtual void SetType(ScalarType type)
    {
      throw new NotImplementedException();
    }

    public virtual (Tensor value, Tensor policy, Tensor mlh, Tensor unc, FP16[] extraStats0, FP16[] extraStats1) forwardValuePolicyMLH_UNC(Tensor inputSquares, Tensor inputMoves)
    {
      return Forward(inputSquares, inputMoves);
    }
  }
}
