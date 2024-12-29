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

using System.Collections.Generic;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp;
using TorchSharp.Modules;

using static CeresTrain.Utils.ModuleParamLoadingUtils;
using Ceres.Base.Misc;
using CeresTrain.Utils;

#endregion

namespace CeresTrain.Networks.Transformer
{
  /// <summary>
  /// Head layer for Ceres Transformer.
  /// 
  /// Based on assumption of 3 linear layers, with activation functions after 2 or 3 of them.
  /// </summary>
  public class NetTransformerLayerHead : Module<Tensor, Tensor, Tensor>
  {
    /// <summary>
    /// Input dimension.
    /// </summary>
    public readonly int InputDim;

    /// <summary>
    /// Type of activation to use for all but final layer.
    /// </summary>
    public NetTransformerDef.ActivationType Activation;

    /// <summary>
    /// Type of activation to use for final layer.
    /// </summary>
    public string FinalActivation;


    /// <summary>
    /// Input width of first linear layer.
    /// </summary>
    public readonly int Dim1;

    /// <summary>
    /// Input width of second linear layer.
    /// </summary>
    public readonly int Dim2;

    /// <summary>
    /// First linear layer.
    /// </summary>
    public readonly Linear Linear1;

    /// <summary>
    /// Second linear layer.
    /// </summary>
    public readonly Linear Linear2;


    bool SaveIntermediateActivations;

    public static Tensor lastOutputTrunk;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="modelDim"></param>
    /// <param name="dim1"></param>
    /// <param name="dim2"></param>
    /// <param name="activation"></param>
    /// <param name="finalActivation"></param>
    /// <param name="saveIntermediateActivations"></param>
    /// <exception cref="NotImplementedException"></exception>
    public NetTransformerLayerHead(string name,
                                   int inputDim,
                                   int dim1, int dim2,
                                   NetTransformerDef.ActivationType activation,
                                   string finalActivation,
                                   bool saveIntermediateActivations) : base(name)
    {
      Activation = activation;
      FinalActivation = finalActivation;
      InputDim = inputDim;
      Dim1 = dim1;
      Dim2 = dim2;
      SaveIntermediateActivations = saveIntermediateActivations;

      if (saveIntermediateActivations)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, "WARNING: SaveIntermediateActivations is enabled for CeresTransformerLayerHead");
      }

      if (activation == NetTransformerDef.ActivationType.SwiGLU)
      {
        throw new Exception("SwiGLU not supported for network heads.");
      }


      Linear1 = Linear(inputDim, dim1, hasBias: true);     
      Linear2 = Linear(dim1, dim2, hasBias: true);

      RegisterComponents();
    }


    public void LoadWeights(Dictionary<string, Tensor> weightsSource, HashSet<string> weightsLoaded, string linearBaseName)
    {
      LinearLoad(weightsSource, weightsLoaded, Linear1, linearBaseName + ".fc.weight", linearBaseName + ".fc.bias");
      LinearLoad(weightsSource, weightsLoaded, Linear2, linearBaseName + ".fcFinal.weight", linearBaseName + ".fcFinal.bias");
    }



    public override Tensor forward(Tensor x, Tensor state)
    {
      using (DisposeScope disposeScopeEval = NewDisposeScope())
      {
        //***** TEMPORARY! SLOWER!!!
        if (SaveIntermediateActivations)
        {
          throw new NotImplementedException();
//              lastOutputTrunk = x.clone().DetachFromDisposeScope();
        }

        Tensor x1 = Linear1.call(x);

        x1 = TorchSharpUtils.WithActivation(x1, Activation);
        Tensor x2 = Linear2.call(x1);
        x1.Dispose();

        if (FinalActivation != null)
        {
          if (FinalActivation == "RELU")
          {
            x2 = functional.relu(x2);
          }
          else
          {
            throw new NotImplementedException();
          }
        }

        return x2.MoveToOuterDisposeScope();
      }
    }

  }

}
