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

using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using CeresTrain.Utils;
using CeresTrain.Networks.MiscModules;
using System;

#endregion

namespace CeresTrain.Networks.Transformer
{
  /// <summary>
  /// Input embedding layer (from raw squares representation to transformer representation dimensions).
  /// </summary>
  internal class NetTransformerLayerEmbedding : Module<Tensor, Tensor>
  {
    /// <summary>
    /// Type of normalization layer.
    /// </summary>
    public readonly NetTransformerDef.NormalizationType NormType;

    /// <summary>
    /// Linear embedding layer.
    /// </summary>
    Linear linear;

    /// <summary>
    /// Normalization layer.
    /// </summary>
    RMSNorm norm;  


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="inputDim"></param>
    /// <param name="modelDim"></param>
    public NetTransformerLayerEmbedding(string name, int inputDim, int modelDim, NetTransformerDef.NormalizationType normType) : base("embedding_layer")
    {
      if (normType != NetTransformerDef.NormalizationType.RMSNorm)
      {
        throw new NotImplementedException("Only RMSNorm currently supported");
      }

      this.name = name;
      NormType = normType;

      norm = new RMSNorm(modelDim, eps: 1e-6f);
      linear = Linear(inputDim, modelDim, hasBias: true);

      RegisterComponents();
    }


    /// <summary>
    /// Loads weights from dictionary of model weights.
    /// </summary>
    /// <param name="weightsSource"></param>
    /// <param name="weightsLoaded"></param>
    public void LoadWeights(Dictionary<string, Tensor> weightsSource, HashSet<string> weightsLoaded)
    {
      ModuleParamLoadingUtils.LinearLoad(weightsSource, weightsLoaded, linear, "embedding_layer.weight", "embedding_layer.bias");
      ModuleParamLoadingUtils.RMSNormLoad(weightsSource, weightsLoaded, norm, "embedding_norm.scale");
    }


    /// <summary>
    /// Forward inference function.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public override Tensor forward(Tensor input) => norm.call(linear.call(input));
  }

}
