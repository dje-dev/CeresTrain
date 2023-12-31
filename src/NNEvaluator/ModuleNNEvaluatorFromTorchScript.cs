﻿#region Using directives

using System;

using TorchSharp;
using static TorchSharp.torch;

using Ceres.Base.DataTypes;
using CeresTrain.Utils;
using CeresTrain.Networks;
using CeresTrain.Networks.Transformer;
using CeresTrain.Trainer;

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
    /// Constructor from given Ceres net definition and underlying net.
    /// </summary>
    /// <param name="netDef"></param>
    /// <param name="execConfig"></param>
    /// <exception cref="Exception"></exception>
    public ModuleNNEvaluatorFromTorchScript(ICeresNeuralNetDef netDef, ConfigNetExecution execConfig)
    {
      CeresNeuralNet transformer = netDef.CreateNetwork(execConfig);
      if (execConfig.TrackFinalLayerIntrinsicDimensionality
        && execConfig.EngineType != NNEvaluatorInferenceEngine.CSharpViaTorchscript)
      {
        throw new Exception("trackFinalIntrinsicDimensionality only supported for CSharpViaTorchscript");
      }

      Device = execConfig.Device;
      DataType = execConfig.DataType;
      CeresNet = transformer;
      TorchscriptFileName1 = execConfig.SaveNetwork1FileName;
    }


    /// <summary>
    /// Constructor from given Ceres net definition.
    /// </summary>
    /// <param name="executionConfig"></param>
    /// <param name="transformerConfig"></param>
    /// <exception cref="Exception"></exception>
    public ModuleNNEvaluatorFromTorchScript(ConfigNetExecution executionConfig, NetTransformerDef transformerConfig)
    {
      if (executionConfig.TrackFinalLayerIntrinsicDimensionality && executionConfig.EngineType != NNEvaluatorInferenceEngine.CSharpViaTorchscript)
      {
        throw new Exception("trackFinalIntrinsicDimensionality only supported for CSharpViaTorchscript");
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

      bool useCSharp = executionConfig.EngineType == NNEvaluatorInferenceEngine.CSharpViaTorchscript;
      if (useCSharp)
      {
        CeresNet = transformerConfig.CreateNetwork(executionConfig);
        CeresNet.eval();
      }
      else
      {
        // NOTE: better to move to device first so type conversion can happen on potentially faster device
        module = TorchscriptUtils.TorchscriptFilesAveraged<Tensor, (Tensor, Tensor, Tensor, Tensor)>(executionConfig.SaveNetwork1FileName, executionConfig.SaveNetwork2FileName, executionConfig.Device, executionConfig.DataType);
        //      module = torch.jit.load<Tensor, (Tensor, Tensor, Tensor, Tensor)>(fileNameTorchscript, Device.type, Device.index).to(dataType);

        //module = TorchscriptUtils. TorchscriptFilesAveraged<Tensor, (Tensor, Tensor, Tensor, Tensor)>(tsFileName1, tsFileName2, device, dataType);
        //      module = module.to(dataType).to(device);
        module.eval();
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

            ret = CeresNet.call(inputSquares);

            NetTransformer ceresTransformer = (NetTransformer)CeresNet as NetTransformer;

            // Save the intrinsic dimensionality of the final layer
            if (ceresTransformer != null)
            {
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
