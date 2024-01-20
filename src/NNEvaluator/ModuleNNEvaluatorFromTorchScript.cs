#region Using directives

using System;

using TorchSharp;
using static TorchSharp.torch;

using Ceres.Base.DataTypes;
using CeresTrain.Utils;
using CeresTrain.Networks;
using CeresTrain.Networks.Transformer;
using CeresTrain.Trainer;
using Ceres.Chess.LC0NetInference;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators;
using Chess.Ceres.NNEvaluators;
using Ceres.Chess.LC0.Batches;
using CeresTrain.TPG;
using System.Runtime.InteropServices;

#endregion

namespace CeresTrain.NNEvaluators
{
  public class ModuleNNEvaluatorFromTorchScript : IModuleNNEvaluator
  {
    /// <summary>
    /// File name of Torchscript file from which model weights are loaded.
    /// </summary>
    public readonly string TorchscriptFileName1;

    /// <summary>
    /// Optional file name of second Torchscript file from which model weights are loaded (for averaging with other file).
    /// </summary>
    public readonly string TorchscriptFileName2;

    /// <summary>
    /// Device onto which to load model.
    /// </summary>
    public readonly Device Device;

    /// <summary>
    /// Data type to use for model.
    /// </summary>
    public readonly ScalarType DataType;

    /// <summary>
    /// Underlying Ceres neural net.
    /// </summary>
    public readonly CeresNeuralNet CeresNet;


    /// <summary>
    /// The loaded Torchscript module.
    /// </summary>
    jit.ScriptModule<Tensor, (Tensor, Tensor, Tensor, Tensor)> module;


    /// <summary>
    /// Constructor from given Ceres net definition.
    /// </summary>
    /// <param name="executionConfig"></param>
    /// <param name="transformerConfig"></param>
    /// <exception cref="Exception"></exception>
    public ModuleNNEvaluatorFromTorchScript(in ConfigNetExecution executionConfig, in NetTransformerDef transformerConfig)
    {
      if (executionConfig.TrackFinalLayerIntrinsicDimensionality && executionConfig.EngineType != NNEvaluatorInferenceEngineType.CSharpViaTorchscript)
      {
        throw new Exception("TrackFinalIntrinsicDimensionality only supported for CSharpViaTorchscript");
      }

      Device = executionConfig.Device;
      TorchscriptFileName1 = executionConfig.SaveNetwork1FileName;
      TorchscriptFileName2 = executionConfig.SaveNetwork2FileName;
      DataType = executionConfig.DataType;

      //Environment.SetEnvironmentVariable("PYTORCH_NVFUSER_DISABLE_FALLBACK", "1");

      Console.WriteLine("LOAD OF TORCHSCRIPT MODEL **********");
      Console.WriteLine("  " + executionConfig.SaveNetwork1FileName);
      if (executionConfig.SaveNetwork1FileName != null)
      {
        Console.WriteLine("  " + executionConfig.SaveNetwork2FileName);
      }

      if (executionConfig.EngineType == NNEvaluatorInferenceEngineType.CSharpViaTorchscript)
      {
        CeresNet = transformerConfig.CreateNetwork(executionConfig);
        CeresNet.eval();
      }
      else if (executionConfig.EngineType == NNEvaluatorInferenceEngineType.TorchViaTorchscript)
      {
        // NOTE: better to move to device first so type conversion can happen on potentially faster device
        module = TorchscriptUtils.TorchScriptFilesAveraged<Tensor, (Tensor, Tensor, Tensor, Tensor)>(executionConfig.SaveNetwork1FileName, executionConfig.SaveNetwork2FileName, executionConfig.Device, executionConfig.DataType);
        //      module = torch.jit.load<Tensor, (Tensor, Tensor, Tensor, Tensor)>(fileNameTorchscript, Device.type, Device.index).to(dataType);

        //module = TorchscriptUtils. TorchscriptFilesAveraged<Tensor, (Tensor, Tensor, Tensor, Tensor)>(tsFileName1, tsFileName2, device, dataType);
        //      module = module.to(dataType).to(device);
        module.eval();
      }
      else
      {
        throw new Exception("Internal error, unexpected inference engine type: " + executionConfig.EngineType);
      } 

    }

    public void SetTraining(bool trainingMode)
    {
      if (trainingMode)
      {
        module.train();
      }
      else
      {
        module.eval();
      }
    }


    /// <summary>
    /// Sets data type of model.
    /// </summary>
    /// <param name="type"></param>
    void IModuleNNEvaluator.SetType(ScalarType type)
    {
      //      this.module.to(type);
    }


    /// <summary>
    /// Returns string description of evaluator.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $">ModuleNNEvaluatorFromTorchScript({TorchscriptFileName1}, {TorchscriptFileName2}, {Device}, {DataType})>";
    }


    /// <summary>
    /// Runs forward pass.
    /// </summary>
    /// <param name="inputSquares"></param>
    /// <param name="inputMoves"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public (Tensor value, Tensor policy, Tensor mlh, Tensor unc, FP16[] extraStats0, FP16[] extraStats1) forwardValuePolicyMLH_UNC(Tensor inputSquares, Tensor inputMoves)
    {
      if (inputMoves is not null)
      {
        throw new Exception("inputMoves not supported");
      }

      FP16[] extraStats0 = null;

      DisposeScope disposeScopeEval;
      using (disposeScopeEval = NewDisposeScope())
      {
        using (no_grad())
        {
          lock (this)
          {
            (Tensor policy1858, Tensor valueWDL, Tensor mlh, Tensor unc) ret = default;

            if (module != null)
            {
              ret = module.call(inputSquares); // N.B. Possible bug, calling this twice sometimes results in slightly different return values
            }
            else
            {
              ret = CeresNet.call(inputSquares);
            }

            // Save the intrinsic dimensionality of the final layer
            if (CeresNet is NetTransformer)
            {
              NetTransformer ceresTransformer = (NetTransformer)CeresNet as NetTransformer;
              extraStats0 = ceresTransformer.IntrinsicDimensionalitiesLastBatch;
            }

            return (ret.valueWDL.MoveToOuterDisposeScope(),
                    ret.policy1858.MoveToOuterDisposeScope(),
                    ret.mlh.MoveToOuterDisposeScope(),
                    ret.unc.MoveToOuterDisposeScope(),
                    extraStats0, null);
          }
        }
      }
    }
  }
}
