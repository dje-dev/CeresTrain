#region Using directives

using System;

using TorchSharp;
using static TorchSharp.torch;

using Ceres.Base.DataTypes;

using CeresTrain.Utils;
using CeresTrain.Networks;
using CeresTrain.Networks.Transformer;
using CeresTrain.Trainer;
using Ceres.Chess.NNEvaluators;

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
    jit.ScriptModule<Tensor, Tensor, Tensor[]> module;
    
    /// <summary>
    /// Interface method that returns Torchscript module.
    /// </summary>
    public jit.ScriptModule<Tensor, Tensor, Tensor[]> Module => module;


    public readonly bool UseState;


    /// <summary>
    /// Constructor from given Ceres net definition.
    /// </summary>
    /// <param name="executionConfig"></param>
    /// <param name="transformerConfig"></param>
    /// <exception cref="Exception"></exception>
    public ModuleNNEvaluatorFromTorchScript(in ConfigNetExecution executionConfig, 
                                            Device device, ScalarType dataType,
                                            bool useState = true)
    {
      if (executionConfig.TrackFinalLayerIntrinsicDimensionality && executionConfig.EngineType != NNEvaluatorInferenceEngineType.CSharpViaTorchscript)
      {
        throw new Exception("TrackFinalIntrinsicDimensionality only supported for CSharpViaTorchscript");
      }

      TorchscriptFileName1 = executionConfig.SaveNetwork1FileName;
      TorchscriptFileName2 = executionConfig.SaveNetwork2FileName;

      Device = device;
      DataType = dataType;  // ScalarType.Float16;
      UseState = useState;

      Console.WriteLine("LOAD OF TORCHSCRIPT MODEL **********");
      Console.WriteLine("  " + executionConfig.SaveNetwork1FileName);
      if (executionConfig.SaveNetwork1FileName != null)
      {
        Console.WriteLine("  " + executionConfig.SaveNetwork2FileName);
      }

      if (executionConfig.EngineType == NNEvaluatorInferenceEngineType.CSharpViaTorchscript)
      {
        throw new Exception("Need remediation to push the creation of the network outside this constructor and keep this class pure");
//        CeresNet = transformerConfig.CreateNetwork(executionConfig);
//        CeresNet.eval();
      }
      else if (executionConfig.EngineType == NNEvaluatorInferenceEngineType.TorchViaTorchscript)
      {
        // NOTE: better to move to device first so type conversion can happen on potentially faster device
        module = TorchscriptUtils.TorchScriptFilesAveraged <Tensor, Tensor, Tensor[]>(executionConfig.SaveNetwork1FileName, executionConfig.SaveNetwork2FileName, executionConfig.Device, executionConfig.DataType);
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
      return $"<TS({TorchscriptFileName1}, {TorchscriptFileName2}, {Device}, {DataType})>";
    }


    /// <summary>
    /// Runs forward pass.
    /// </summary>
    /// <param name="inputSquares"></param>
    /// <param name="priorState"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public (Tensor policy, Tensor value, Tensor mlh, Tensor unc, 
            Tensor value2, Tensor qDeviationLower, Tensor qDeviationUpper,
            Tensor uncertaintyPolicy,
            Tensor action, Tensor boardState, Tensor actionUncertainty,
            FP16[] extraStats0, FP16[] extraStats1) forwardValuePolicyMLH_UNC((Tensor squares, Tensor priorState) input)
    {
      FP16[] extraStats0 = null;

      DisposeScope disposeScopeEval;
      using (disposeScopeEval = NewDisposeScope())
      {
        using (no_grad())
        {
          lock (this)
          {
            (Tensor policy1858, Tensor valueWDL, Tensor mlh, Tensor unc, Tensor value2, 
             Tensor qDeviationLower, Tensor qDeviationUpper, Tensor uncertaintyPolicy,
             Tensor actions, Tensor boardState, Tensor actionUncertainty) ret = default;

            bool hasAction;
            bool networkExpectsBoardState = UseState;
            if (module != null)
            {
              Tensor[] rawRet = default;
              const bool ALWAYS_USE_STATE = true; // dummy value required even if not present in net
              if (UseState || ALWAYS_USE_STATE)
              {
                Tensor priorState;
                if (input.priorState is not null)
                {
                  priorState = input.priorState;
                }
                else
                {
                  // TODO: this allocation on every call is expensive, someday allow passing of None to the neural network instead?
                  priorState = torch.zeros([input.squares.shape[0], 64, NNEvaluator.SIZE_STATE_PER_SQUARE], dtype: DataType, device: Device);
                }

                rawRet = module.forward(input.squares, priorState); // N.B. Possible bug, calling this twice sometimes results in slightly different return values
//                var xx = module.forward(new object[] { input.squares, priorState }); // N.B. Possible bug, calling this twice sometimes results in slightly different return values
              }
              else
              {
                object rawRetObj = module.forward(input.squares);
                rawRet = (Tensor[])rawRetObj;
              }

              ret = (rawRet[0], rawRet[1], rawRet[2], rawRet[3], rawRet[4], rawRet[5], rawRet[6],
                     rawRet.Length > 7 ? rawRet[7] : default,
                     rawRet.Length > 8 ? rawRet[8] : default,
                     rawRet.Length > 9 ? rawRet[9] : default,
                     rawRet.Length > 10 ? rawRet[10] : default);
              hasAction = rawRet.Length > 8;
            }
            else
            {
              throw new Exception("Needs remediation for addition of action head and board state");
              //ret = CeresNet.call((input.squares, input.priorState));
            }

            // Save the intrinsic dimensionality of the final layer
            if (CeresNet is NetTransformer)
            {
              NetTransformer ceresTransformer = (NetTransformer)CeresNet as NetTransformer;
              extraStats0 = ceresTransformer.IntrinsicDimensionalitiesLastBatch;
            }

            return (ret.policy1858.MoveToOuterDisposeScope(),
                    ret.valueWDL.MoveToOuterDisposeScope(),
                    ret.mlh.MoveToOuterDisposeScope(),
                    ret.unc.MoveToOuterDisposeScope(),
                    ret.value2.MoveToOuterDisposeScope(),
                    (object)ret.qDeviationLower == null ? null : ret.qDeviationLower.MoveToOuterDisposeScope(),
                    (object)ret.qDeviationUpper == null ? null : ret.qDeviationUpper.MoveToOuterDisposeScope(),
                    null, // TODO: implement policy uncertainty 
                    hasAction ? ret.actions.MoveToOuterDisposeScope() : null,
                    (networkExpectsBoardState && (object)ret.boardState != null) ? ret.boardState.MoveToOuterDisposeScope() : null,
                    null, // TODO: implement action uncertainty
                    extraStats0, null);
          }
        }
      }
    }
  }
}
