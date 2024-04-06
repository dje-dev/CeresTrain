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
  public abstract class CeresNeuralNet : Module<(Tensor squares, Tensor priorState), (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)>, IModuleNNEvaluator
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
    public abstract (Tensor value, Tensor policy, Tensor mlh, Tensor unc, 
                     Tensor value2, Tensor qDeviationLower, Tensor qDeviationUpper,
                     Tensor action, Tensor boardState,
                     FP16[] extraStats0, FP16[] extraStats1) Forward((Tensor squares, Tensor priorState) inputs);


    public virtual void SetType(ScalarType type)
    {
      throw new NotImplementedException();
    }

    public virtual (Tensor value, Tensor policy, Tensor mlh, Tensor unc,
                    Tensor value2, Tensor qDeviationLower, Tensor qDeviationUpper,
                    Tensor action, Tensor boardState,
                    FP16[] extraStats0, FP16[] extraStats1) forwardValuePolicyMLH_UNC((Tensor squares, Tensor priorState) inputs)
    {
      return Forward((inputs.squares, inputs.priorState));
    }
  }
}
