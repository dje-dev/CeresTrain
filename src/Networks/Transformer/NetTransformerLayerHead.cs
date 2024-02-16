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
    /// Input model dimension (embedding dimension).
    /// </summary>
    public readonly int ModelDim;

    /// <summary>
    /// Width of global stream (if any).
    /// </summary>
    public readonly int GlobalStreamDim;

    /// <summary>
    /// Number of input square (sequence length).
    /// </summary>
    public readonly int NumSquares;

    /// <summary>
    /// Type of activation to use for all but final layer.
    /// </summary>
    public NetTransformerDef.ActivationType Activation;

    /// <summary>
    /// Type of activation to use for final layer.
    /// </summary>
    public string FinalActivation;


    /// <summary>
    /// Divisor used to convert between embedding dimension and reduced dimension output by reduce squares layer.
    /// </summary>
    public int PremapDimDivisor;

    /// <summary>
    /// Input width of first linear layer.
    /// </summary>
    public readonly int Dim1;

    /// <summary>
    /// Input width of second linear layer.
    /// </summary>
    public readonly int Dim2;

    /// <summary>
    /// Input width of third linear layer.
    /// </summary>
    public readonly int Dim3;

    /// <summary>
    /// Optional initial layer that maps vectors from each square
    /// to a smaller dimension before concatenation.
    /// </summary>
    public Linear LinearPremap;

    /// <summary>
    /// First linear layer.
    /// </summary>
    public readonly Linear Linear1;

    /// <summary>
    /// Second linear layer.
    /// </summary>
    public readonly Linear Linear2;

    /// <summary>
    /// Final linear layer.
    /// </summary>
    public readonly Linear Linear3;

    bool SaveIntermediateActivations;

    bool PremapAlreadyApplied;

    public static Tensor lastOutputTrunk;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="numSquares"></param>
    /// <param name="modelDim"></param>
    /// <param name="dimGlobalStream"></param>
    /// <param name="premapDimDivisor"></param>
    /// <param name="dim1"></param>
    /// <param name="dim2"></param>
    /// <param name="dim3"></param>
    /// <param name="activation"></param>
    /// <param name="finalActivation"></param>
    /// <param name="saveIntermediateActivations"></param>
    /// <param name="reduceSquaresAlreadyApplied"></param>
    /// <exception cref="NotImplementedException"></exception>
    public NetTransformerLayerHead(string name, int numSquares,
                                   int modelDim,
                                   int dimGlobalStream,
                                   int premapDimDivisor,
                                   int dim1, int dim2, int dim3,
                                   NetTransformerDef.ActivationType activation,
                                   string finalActivation,
                                   bool saveIntermediateActivations,
                                   bool reduceSquaresAlreadyApplied) : base(name)
    {
      NumSquares = numSquares;
      Activation = activation;
      FinalActivation = finalActivation;
      ModelDim = modelDim;
      GlobalStreamDim = dimGlobalStream; ;
      PremapDimDivisor = premapDimDivisor;
      Dim1 = dim1;
      Dim2 = dim2;
      Dim3 = dim3;
      SaveIntermediateActivations = saveIntermediateActivations;
      PremapAlreadyApplied = reduceSquaresAlreadyApplied;

      if (saveIntermediateActivations)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, "WARNING: SaveIntermediateActivations is enabled for CeresTransformerLayerHead");
      }

      if (activation == NetTransformerDef.ActivationType.SwiGLU)
      {
        throw new Exception("SwiGLU not supported for network heads.");
      }

      if (!reduceSquaresAlreadyApplied)
      {
        if (premapDimDivisor > 1)
        {
          LinearPremap = Linear(modelDim, modelDim / premapDimDivisor, hasBias: true);
        }
      }

      Linear1 = Linear(GlobalStreamDim + NumSquares * modelDim / premapDimDivisor, dim1, hasBias: true);
      Linear2 = Linear(dim1, dim2, hasBias: true);
      Linear3 = Linear(dim2, dim3, hasBias: true);

      RegisterComponents();
    }


    public void LoadWeights(Dictionary<string, Tensor> weightsSource,
                            HashSet<string> weightsLoaded,
                            string premapLayerName, string linearBaseName)
    {
      if (PremapDimDivisor > 1)
      {
        LinearLoad(weightsSource, weightsLoaded, LinearPremap, premapLayerName + ".weight", premapLayerName + ".bias");
      }

      LinearLoad(weightsSource, weightsLoaded, Linear1, linearBaseName + "1.weight", linearBaseName + "1.bias");
      LinearLoad(weightsSource, weightsLoaded, Linear2, linearBaseName + "2.weight", linearBaseName + "2.bias");
      LinearLoad(weightsSource, weightsLoaded, Linear3, linearBaseName + "3.weight", linearBaseName + "3.bias");
    }



    public override Tensor forward(Tensor x, Tensor state)
    {
      using (DisposeScope disposeScopeEval = NewDisposeScope())
      {
        if (!PremapAlreadyApplied)
        {
          if (PremapDimDivisor > 1)
          {
            x = LinearPremap.call(x);

            //***** TEMPORARY! SLOWER!!!
            if (SaveIntermediateActivations)
            {
              lastOutputTrunk = x.clone().DetachFromDisposeScope();
            }
          }
        }

        x = x.reshape(-1,  NumSquares * (ModelDim / PremapDimDivisor));

        if (GlobalStreamDim > 0)
        {
          x = torch.concat([x, state], 1);
        }

        Tensor x1 = Linear1.call(x);
        x.Dispose();

        x1 = TorchSharpUtils.WithActivation(x1, Activation);
        Tensor x2 = Linear2.call(x1);
        x1.Dispose();
        x2 = TorchSharpUtils.WithActivation(x2, Activation); 

        Tensor x3 = Linear3.call(x2);

        if (FinalActivation != null)
        {
          if (FinalActivation == "RELU")
          {
            x3 = functional.relu(x3);
          }
          else
          {
            throw new NotImplementedException();
          }
        }

        return x3.MoveToOuterDisposeScope();
      }
    }

  }

}
