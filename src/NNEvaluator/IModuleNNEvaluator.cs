﻿#region Using directives

using System;
using Ceres.Base.DataTypes;
using static TorchSharp.torch;

#endregion

namespace CeresTrain.NNEvaluators
{
  /// <summary>
  /// Interface implemented by any neural network evaluator.
  /// </summary>
  public interface IModuleNNEvaluator
  {
    void SetTraining(bool trainingMode) => throw new NotImplementedException();
    void SetType(ScalarType type) => throw new NotImplementedException();
    (Tensor value, Tensor policy, Tensor mlh, Tensor unc, FP16[] extraStats0, FP16[] extraStats1) forwardValuePolicyMLH_UNC(Tensor inputSquares, Tensor inputMoves);
  }

}
