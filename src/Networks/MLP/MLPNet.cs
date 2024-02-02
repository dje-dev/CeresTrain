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
using TorchSharp.Modules;
using static TorchSharp.torch.nn;

using CeresTrain.Trainer;
using Ceres.Base.DataTypes;
using CeresTrain.TPG;

#endregion

namespace CeresTrain.Networks.MLP
{
  /// <summary>
  /// Ceres neural network consisting only of linear layers.
  /// 
  /// TODO: Generalize to e other than fixed 4 layers, support LayerNorm.
  /// </summary>

  public class MLPNet : CeresNeuralNet
  {
    /// <summary>
    /// 
    /// </summary>
    public readonly MLPNetDef NetDef;


    /// <summary>
    /// Execution configuration for the network.
    /// </summary>
    public readonly ConfigNetExecution ExecutionConfig;


    private Sequential model;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="netDef"></param>
    /// <param name="executionConfig"></param>
    public MLPNet(in MLPNetDef netDef, in ConfigNetExecution executionConfig)
      : base("CeresNetLinear")
    {
      NetDef = netDef;
      ExecutionConfig = executionConfig;

      ModuleList<Module<Tensor, Tensor>> layers = new ModuleList<Module<Tensor, Tensor>>();

      int thisLayerInputSize = 64 * TPGRecord.BYTES_PER_SQUARE_RECORD;
      for (int i = 0; i < netDef.Layers.Length; i++)
      {
        int outputSize = netDef.Layers[i].width;
        layers.append(Linear(thisLayerInputSize, outputSize));
        layers.append(netDef.Layers[i].activation);
        thisLayerInputSize = outputSize;
      }

      model = Sequential(layers);

      RegisterComponents();
    }


    public override (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) forward(Tensor input)
    {
      var batchSize = input.shape[0];
      Tensor reshaped = input.to(ExecutionConfig.DataType).reshape(batchSize, 64 * TPGRecord.BYTES_PER_SQUARE_RECORD);

      Tensor trunkOutput = model.call(reshaped);

      Tensor value = zeros([batchSize, 3], ExecutionConfig.DataType, ExecutionConfig.Device, requires_grad: true);
      Tensor policy = zeros([batchSize, 1858], ExecutionConfig.DataType, ExecutionConfig.Device, requires_grad: true);
      Tensor mlh = zeros([batchSize, 1], ExecutionConfig.DataType, ExecutionConfig.Device, requires_grad: true);
      Tensor unc = zeros([batchSize, 1], ExecutionConfig.DataType, ExecutionConfig.Device, requires_grad: true);
      Tensor value2 = zeros([batchSize, 3], ExecutionConfig.DataType, ExecutionConfig.Device, requires_grad: true);
      Tensor qDeviationLower = zeros([batchSize, 1], ExecutionConfig.DataType, ExecutionConfig.Device, requires_grad: true);
      Tensor qDeviationUpper = zeros([batchSize, 1], ExecutionConfig.DataType, ExecutionConfig.Device, requires_grad: true);

      return (value, policy, mlh, unc, value2, qDeviationLower, qDeviationUpper);
    }

    public override (Tensor value, Tensor policy, Tensor mlh, Tensor unc, 
                     Tensor value2, Tensor qDeviationLower, Tensor qDeviationUpper,
                     FP16[] extraStats0, FP16[] extraStats1) Forward(Tensor inputSquares, Tensor inputMoves)
    {
      (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) ret = forward(inputSquares);
      return (ret.Item1, ret.Item2, ret.Item3, ret.Item4, ret.Item5, ret.Item6, ret.Item7, null, null);
    }
  }

}
