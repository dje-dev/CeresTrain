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
using CeresTrain.Networks.MiscModules;
using static CeresTrain.Networks.Transformer.NetTransformer;

#endregion

namespace CeresTrain.Networks.Transformer
{
  /// <summary>
  /// Head layer for Ceres Transformer.
  /// 
  /// Based on assumption of 3 linear layers, with activation functions after 2 or 3 of them.
  /// </summary>
  public class NetTransformerLayerHead : Module<Tensor, Tensor>
  {

    /// <summary>
    /// Transformer to which this head belongs.
    /// </summary>
    public readonly NetTransformer Parent;

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
    public readonly Module<Tensor,Tensor> Linear1;

    /// <summary>
    /// Second linear layer.
    /// </summary>
    public readonly Module<Tensor, Tensor> Linear2;


    bool SaveIntermediateActivations;

    public static Tensor lastOutputTrunk;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="parent"></param>
    /// <param name="name"></param>
    /// <param name="inputDim"></param>
    /// <param name="dim1"></param>
    /// <param name="dim2"></param>
    /// <param name="activation"></param>
    /// <param name="finalActivation"></param>
    /// <param name="saveIntermediateActivations"></param>
    /// <exception cref="Exception"></exception>
    public NetTransformerLayerHead(NetTransformer parent,
                                   string name,
                                   int inputDim,
                                   int dim1, int dim2,
                                   NetTransformerDef.ActivationType activation,
                                   string finalActivation,
                                   bool saveIntermediateActivations) : base(name)
    {
      Parent = parent;
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
      Linear1 = LoRALinear.PossiblyLoRAWrappedModule(Linear1, Parent.TransformerConfig.LoRARankDivisor,
                                                  () => Parent.LoRAEnabled, Parent.EligibleForLoRA(0, LayerTypeEnum.Head));

      Linear2 = Linear(dim1, dim2, hasBias: true);
      Linear2 = LoRALinear.PossiblyLoRAWrappedModule(Linear2, Parent.TransformerConfig.LoRARankDivisor,
                                                  () => Parent.LoRAEnabled, Parent.EligibleForLoRA(0, LayerTypeEnum.Head));

      RegisterComponents();
    }


    public void LoadWeights(Dictionary<string, Tensor> weightsSource, HashSet<string> weightsLoaded, string linearBaseName)
    {
      LinearLoad(weightsSource, weightsLoaded, LoRALinear.BaseLinear(Linear1), linearBaseName + ".fc.weight", linearBaseName + ".fc.bias");
      LinearLoad(weightsSource, weightsLoaded, LoRALinear.BaseLinear(Linear2), linearBaseName + ".fcFinal.weight", linearBaseName + ".fcFinal.bias");
    }



    public override Tensor forward(Tensor x)
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
