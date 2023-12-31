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

using CeresTrain.Networks;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Interface implemented by classes/structs/records which 
  /// define a neural network of a particular type,
  /// implementing a factory method to instantiate an 
  /// actual CeresNeuralNetwork network from the definition.
  /// </summary>
  public interface ICeresNeuralNetDef
  {
    /// <summary>
    /// Factory method to create an actual neural network from the definition.
    /// </summary>
    /// <param name="netConfig"></param>
    /// <returns></returns>
    CeresNeuralNet CreateNetwork(in ConfigNetExecution netConfig);

    /// <summary>
    /// Check if the configuration is valid.
    /// </summary>
    void Validate();
  }
}
