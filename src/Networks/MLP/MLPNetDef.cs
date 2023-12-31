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

using static TorchSharp.torch;
using CeresTrain.Trainer;

#endregion

namespace CeresTrain.Networks.MLP
{
  /// <summary>
  /// Definition of a neural network consisting of a sequence of linear layers
  /// separated by RELU non-linearities.
  /// </summary>
  public class MLPNetDef : ICeresNeuralNetDef
  {
    /// <summary>
    /// Sequence of linear layers and their widths/activations.
    /// </summary>
    public (int width, nn.Module<Tensor, Tensor> activation)[] Layers;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="layers"></param>
    public MLPNetDef(params (int width, nn.Module<Tensor, Tensor> activation)[] layers)
    {
      Layers = layers;
    }


    /// <summary>
    /// Factory method to create an actual neural network from the definition.
    /// </summary>
    /// <param name="netConfig"></param>
    /// <returns></returns>
    public CeresNeuralNet CreateNetwork(in ConfigNetExecution executionConfig)
    {
      return new MLPNet(this, executionConfig);
    }

    /// <summary>
    /// Check if the configuration is valid.
    /// </summary>
    public void Validate()
    {

    }
  }
}
