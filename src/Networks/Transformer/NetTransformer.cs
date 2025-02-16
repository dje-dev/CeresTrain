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

using System.IO;
using System.Collections.Generic;
using System.Linq;

using TorchSharp;
using static TorchSharp.torch.jit;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

using Ceres.Base.Misc;
using Ceres.Base.DataTypes;
using Ceres.Chess.NNEvaluators.Ceres.TPG;

using static CeresTrain.Utils.ModuleParamLoadingUtils;
using CeresTrain.NNIntrospection;
using CeresTrain.Utils;
using CeresTrain.Trainer;
using CeresTrain.Networks.MiscModules;

#endregion

namespace CeresTrain.Networks.Transformer
{
  /// <summary>
  /// Full Ceres Transformer model (embedding, encoder layer, heads).
  /// </summary>
  public class NetTransformer : CeresNeuralNet
  {
    public const float MLH_DIVISOR = 100;

    public static bool DUMP_INFO_TO_CONSOLE = false;

    public NetTransformerDef TransformerConfig;
    public ConfigNetExecution ExecutionConfig;

    /// <summary>
    /// Network can be configured to run in one of two modes:
    ///   - LoraEnabled is true, most layers frozen and LoRA adaptation layers enabled
    ///   - LoraEnabled is false, normal mode and any possibly attached LoRA adaptations are not included
    /// </summary>
    public bool LoRAEnabled;

    public ParameterStats ParameterStats => new(this);


    /// <summary>
    /// Debug comparison network against which the output of the DebugCompareLayerName will be compared.
    /// </summary>
    public Module DebugComparisonNetwork = default;

    /// <summary>
    /// Name of the layer in the DebugComparisonNetwork against which the output from this network 
    /// will be compared (in DebugCompareNetworkOutputs method).
    /// </summary>
    public string DebugCompareLayerName;


    // First layer of heads projects size down from embedding dimension by some factor
    // (after which all the independent squares have their outputs concatenated together).
    const int HEAD_PREMAP_DIVISOR = 16;


    internal NetTransformerLayerEmbedding layerEmbedding;
    internal Linear globalStateEmbedding;

    internal NetTransformerLayerEncoder[] layersEncodersArray;

    internal Module<Tensor,Tensor> headPremap;
    internal Module<Tensor,Tensor> headSharedLinear;

    public Module<Tensor,Tensor> layerValueHead;
    internal NetTransformerLayerHead layerValue2Head;
    internal NetTransformerLayerHead layerPolicyHead;
    internal NetTransformerLayerHead layerMLHHead;
    internal NetTransformerLayerHead layerUNCHead;
    internal NetTransformerLayerHead layerUncPolicyHead;
    internal NetTransformerLayerHead layerQDeviationLowerHead;
    internal NetTransformerLayerHead layerQDeviationUpperHead;

    Linear smLinearShared = null;

    public string LayerNameToTrackIntrinsicDimensionality = null;
    public FP16[] IntrinsicDimensionalitiesLastBatch = null;


    /// <summary>
    /// Constructor for a CeresTransformer network with specified configuration.
    /// </summary>
    /// <param name="transformerNetDef"></param>
    /// <param name="device"></param>
    /// <param name="overrideWeights"></param>
    public NetTransformer(ConfigNetExecution executionConfig,
                          NetTransformerDef transformerNetDef,
                          Dictionary<string, Tensor> overrideWeights = null)
        : base("CeresTransformer")
    {
      TransformerConfig = transformerNetDef;
      ExecutionConfig = executionConfig;

      if (TransformerConfig.TrainOn4BoardSequences)
        throw new NotImplementedException("TrainOn4BoardSequences not implemented");

      float alpha = transformerNetDef.DeepNorm ? MathF.Pow(2 * TransformerConfig.NumLayers, 0.25f) : 1;

      if (DUMP_INFO_TO_CONSOLE)
      {
        Console.WriteLine("LOADING TORCHSCRIPT MODEL  "
                          + executionConfig.SaveNetwork1FileName + " "
                          + executionConfig.SaveNetwork2FileName);
      }

      // 1) Figure out paramsToLoad from overrideWeights or from .ts file (if needed)
      Dictionary<string, Tensor> paramsToLoad = null;

      if (overrideWeights != null)
      {
        paramsToLoad = overrideWeights;
      }
      else if (ExecutionConfig.SaveNetwork1FileName != null &&
              (!ExecutionConfig.SaveNetwork1FileName.ToLower().EndsWith(".ts") &&
               !ExecutionConfig.SaveNetwork1FileName.ToLower().EndsWith(".dat")))
      {
        throw new Exception("Specified ExecutionConfig.SaveNetwork1FileName must end with either .ts or .dat, instead see: "
                            + ExecutionConfig.SaveNetwork1FileName);
      }
      else if (ExecutionConfig.SaveNetwork1FileName != null &&
               !File.Exists(ExecutionConfig.SaveNetwork1FileName))
      {
        throw new Exception("Specified ExecutionConfig.SaveNetwork1FileName does not exist: "
                            + ExecutionConfig.SaveNetwork1FileName);
      }
      else if (ExecutionConfig.SaveNetwork1FileName != null &&
               ExecutionConfig.SaveNetwork1FileName.ToLower().EndsWith(".ts"))
      {
        // Load averaged TorchScript
        ScriptModule<Tensor, Tensor, (Tensor, Tensor, Tensor, Tensor)> transformerTS
            = TorchscriptUtils.TorchScriptFilesAveraged<Tensor, Tensor, (Tensor, Tensor, Tensor, Tensor)>(
                ExecutionConfig.SaveNetwork1FileName,
                ExecutionConfig.SaveNetwork2FileName,
                ExecutionConfig.Device,
                ExecutionConfig.DataType);

        paramsToLoad = new Dictionary<string, Tensor>();
        transformerTS.named_parameters().ToList().ForEach(p => paramsToLoad.Add(p.name, p.parameter));
      }

      // 2) Build modules
      // EMBEDDING
      int embeddingWidth = TransformerConfig.ModelDim;
      layerEmbedding = new NetTransformerLayerEmbedding("embedding",
                                                        TPGRecord.BYTES_PER_SQUARE_RECORD,
                                                        embeddingWidth,
                                                        transformerNetDef.NormType);
      layerEmbedding = layerEmbedding.to(ExecutionConfig.Device, ExecutionConfig.DataType);

      // DENSEFORMER (not yet supported)
      if (TransformerConfig.DenseFormer)
        throw new NotImplementedException("DenseFormer not implemented");

      if (TransformerConfig.SmolgenDimPerSquare > 0)
      {
        smLinearShared = Linear(TransformerConfig.SmolgenDim / TransformerConfig.SmolgenToHeadDivisor,
                                64 * 64,
                                true,
                                ExecutionConfig.Device,
                                ExecutionConfig.DataType);
      }

      // ENCODER LAYERS
      layersEncodersArray = new NetTransformerLayerEncoder[TransformerConfig.NumLayers];

      for (int layerNum = 0; layerNum < TransformerConfig.NumLayers; layerNum++)
      {
        bool hasDual = TransformerConfig.DualAttentionMode != NetTransformerDef.DualAttentionModeType.None
                       && paramsToLoad != null
                       && paramsToLoad.ContainsKey($"transformer_layer.{layerNum}.attention2.qkv.weight");// sometimes dual only present in certain layers, e.g. every other one
        NetTransformerDef.DualAttentionModeType dualMode = !hasDual
            ? NetTransformerDef.DualAttentionModeType.None
            : TransformerConfig.DualAttentionMode;

        NetTransformerLayerEncoder teCS = new(
            this,
            TransformerConfig.NumLayers,
            layerNum,
            TransformerConfig.ModelDim,
            TransformerConfig.NumHeads,
            TransformerConfig.PreNorm,
            TransformerConfig.NormType,
            TransformerConfig.AttentionMultiplier,
            dualMode,
            TransformerConfig.NonLinearAttention,
            TransformerConfig.FFNMultiplier,
            TransformerConfig.FFNActivationType,
            alpha,
            ExecutionConfig.DropoutRate,
            ExecutionConfig.DropoutDuringInference,
            TransformerConfig.SoftCapThreshold,
            ExecutionConfig.SupplementaryStat == NNLayerMonitor.SupplementaryStatType.AverageCosineSimilarity,
            TransformerConfig.SmolgenDimPerSquare,
            TransformerConfig.SmolgenDim,
            TransformerConfig.SmolgenToHeadDivisor,
            TransformerConfig.SmolgenActivationType,
            TransformerConfig.SoftMoEConfig,
            ExecutionConfig.MonitorActivationStats,
            ref smLinearShared,
            TransformerConfig.LoRARankDivisor,
            () => LoRAEnabled,
            TransformerConfig.FFNUseGlobalEveryNLayers > 0 && (layerNum % TransformerConfig.FFNUseGlobalEveryNLayers)
                                                               == TransformerConfig.FFNUseGlobalEveryNLayers - 1
        );
        teCS = teCS.to(ExecutionConfig.Device, ExecutionConfig.DataType);
        layersEncodersArray[layerNum] = teCS;
        register_module($"encoder_layer_{layerNum}", teCS);
      }

      // Head premap
      headPremap = Linear(TransformerConfig.ModelDim,
                          TransformerConfig.ModelDim / HEAD_PREMAP_DIVISOR,
                          true,
                          ExecutionConfig.Device,
                          ExecutionConfig.DataType);
      headPremap = LoRALinear.PossiblyLoRAWrappedModule(headPremap, TransformerConfig.LoRARankDivisor, () => LoRAEnabled);
      headPremap.to(ExecutionConfig.Device, ExecutionConfig.DataType);

      // Head shared linear
      headSharedLinear = Linear(64 * TransformerConfig.ModelDim / HEAD_PREMAP_DIVISOR,
                                TransformerConfig.ModelDim,
                                true,
                                ExecutionConfig.Device,
                                ExecutionConfig.DataType);
      headSharedLinear = LoRALinear.PossiblyLoRAWrappedModule(headSharedLinear, TransformerConfig.LoRARankDivisor, () => LoRAEnabled);
      headSharedLinear.to(ExecutionConfig.Device, ExecutionConfig.DataType);

      // Optional MLH head
      if (paramsToLoad != null && paramsToLoad.ContainsKey("mlh_head.fc.weight"))
      {
        layerMLHHead = new NetTransformerLayerHead(this, "mlh_head", 512, 128, 1,
                                                   TransformerConfig.HeadsActivationType,
                                                   null, false);
        layerMLHHead.to(ExecutionConfig.Device, ExecutionConfig.DataType);
      }

      // UNC head
      layerUNCHead = new NetTransformerLayerHead(this, "unc_head",
                                                 TransformerConfig.ModelDim, 128, 1,
                                                 TransformerConfig.HeadsActivationType, null, false);
      layerUNCHead.to(ExecutionConfig.Device, ExecutionConfig.DataType);

      // UNC policy
      layerUncPolicyHead = new NetTransformerLayerHead(this, "unc_policy",
                                                       TransformerConfig.ModelDim, 128, 1,
                                                       TransformerConfig.HeadsActivationType, null, false);
      layerUncPolicyHead.to(ExecutionConfig.Device, ExecutionConfig.DataType);

      // QDev heads
      layerQDeviationLowerHead = new NetTransformerLayerHead(this, "qdev_lower_head",
                                                             TransformerConfig.ModelDim, 128, 1,
                                                             TransformerConfig.HeadsActivationType, null, false);
      layerQDeviationLowerHead.to(ExecutionConfig.Device, ExecutionConfig.DataType);

      layerQDeviationUpperHead = new NetTransformerLayerHead(this, "qdev_upper_head",
                                                             TransformerConfig.ModelDim, 128, 1,
                                                             TransformerConfig.HeadsActivationType, null, false);
      layerQDeviationUpperHead.to(ExecutionConfig.Device, ExecutionConfig.DataType);

      // POLICY HEAD
      layerPolicyHead = new NetTransformerLayerHead(this, "policy_head",
                                                    TransformerConfig.ModelDim, 512, 1858,
                                                    TransformerConfig.HeadsActivationType, null, false);
      layerPolicyHead.to(ExecutionConfig.Device, ExecutionConfig.DataType);

      // VALUE HEAD
      const bool SAVE_INTERMEDIATE_FOR_VALUE_HEAD = false;
      layerValueHead = new NetTransformerLayerHead(this, "value_head",
                                                   TransformerConfig.ModelDim, 256, 3,
                                                   TransformerConfig.HeadsActivationType, null,
                                                   SAVE_INTERMEDIATE_FOR_VALUE_HEAD);
      layerValueHead.to(ExecutionConfig.Device, ExecutionConfig.DataType);

      layerValue2Head = new NetTransformerLayerHead(this, "value2_head",
                                                    TransformerConfig.ModelDim + 2, 256, 3,
                                                    TransformerConfig.HeadsActivationType, null,
                                                    SAVE_INTERMEDIATE_FOR_VALUE_HEAD);
      layerValue2Head.to(ExecutionConfig.Device, ExecutionConfig.DataType);

      // 3) Register everything
      RegisterComponents();

      // 4) Now load all dictionary-based weights (if any) in a separate method
      LoadWeightsIfApplicable(paramsToLoad);



      // 5) If there's a .dat file, load it via the TorchSharp model loader
      if (overrideWeights == null &&
          ExecutionConfig.SaveNetwork1FileName != null &&
          executionConfig.SaveNetwork1FileName.ToLower().EndsWith(".dat"))
      {
        this.to(ExecutionConfig.Device, ExecutionConfig.DataType);
        try
        {
          load(ExecutionConfig.SaveNetwork1FileName);
        }
        catch (Exception e)
        {
          Console.WriteLine("Model named parameters:");
          foreach ((string name, Parameter parameter) node in named_parameters())
          {
            Console.WriteLine("  " + node);
          }
          Console.WriteLine();
          ConsoleUtils.WriteLineColored(ConsoleColor.Red,
              "Error loading Torchscript model, possible mismatch of C# model definition and saved model: "
              + ExecutionConfig.SaveNetwork1FileName
              + " or alternately possibly specified config does not match config used when net was trained.");
          Console.WriteLine($"See list above of expected {named_parameters().Count()} named parameters in Torchscript file based on this C# model");
          throw;
        }
      }

      // Default mode is evaluation (inference).
      eval();

      const bool DUMP_WEIGHT_STATS = false;
      if (DUMP_WEIGHT_STATS)
      {
        ParameterStats.DumpAllStats();
      }

      if (ExecutionConfig.TrackFinalLayerIntrinsicDimensionality)
      {
        string layerName = $"layersEncoders.{transformerNetDef.NumLayers - 1}.mlpLinear2";
        LayerNameToTrackIntrinsicDimensionality = layerName;
      }

      if (LayerNameToTrackIntrinsicDimensionality != null)
      {
        foreach ((string name, Module<Tensor, Tensor> module) node in named_modules())
        {
          if (LayerNameToTrackIntrinsicDimensionality == node.name)
          {
            // Hook to monitor the layer
            node.module.register_forward_hook((mod, input, output) =>
            {
              IntrinsicDimensionalitiesLastBatch = FP16.ToFP16(IntrinsicDimensionality.TwoNN(output));
              return output;
            });
          }
        }
      }

      const int MONITOR_TO_DUMP_RATIO = 20;
      int monitorSkipCount = Math.Max(1, ExecutionConfig.ActivationMonitorDumpSkipCount / MONITOR_TO_DUMP_RATIO);
      NNLayerMonitorSet monitor = ExecutionConfig.ActivationMonitorDumpSkipCount > 0
                                      ? new NNLayerMonitorSet(this, monitorSkipCount,
                                                              ExecutionConfig.ActivationMonitorDumpSkipCount,
                                                              layerName => true)
                                      : null;

      if (ExecutionConfig.SupplementaryStat != NNLayerMonitor.SupplementaryStatType.None)
      {
        // e.g. set up supplemental stats for each encoder
        for (int i = 0; i < TransformerConfig.NumLayers; i++)
        {
          monitor?.SetSupplementalLayerStat($"encoder_layer_{i}", ExecutionConfig.SupplementaryStat);
        }
      }

      if (TransformerConfig.LoRARankDivisor > 0)
      {
        PrepareForLoRATraining();
      }
    }


    /// <summary>
    /// Refactored method to apply weights from the paramsToLoad dictionary,
    /// now called immediately after RegisterComponents().
    /// </summary>
    private void LoadWeightsIfApplicable(Dictionary<string, Tensor> paramsToLoad)
    {
      // If there's no dictionary to load, just return
      if (paramsToLoad == null)
      {
        return;
      }

      using (torch.no_grad())
      {
        HashSet<string> loadedParams = new();

        // 1) EMBEDDING
        layerEmbedding.LoadWeights(paramsToLoad, loadedParams);

        // 2) ENCODER LAYERS
        for (int layerNum = 0; layerNum < TransformerConfig.NumLayers; layerNum++)
        {
          layersEncodersArray[layerNum].LoadWeights(paramsToLoad, loadedParams);
        }

        // 3) Head premap
        LinearLoad(paramsToLoad, loadedParams, LoRALinear.BaseLinear(headPremap), "headPremap.weight", "headPremap.bias");

        // 4) Head shared linear
        LinearLoad(paramsToLoad, loadedParams, LoRALinear.BaseLinear(headSharedLinear), "headSharedLinear.weight", "headSharedLinear.bias");

        // 5) Optional MLH head
        if (paramsToLoad.ContainsKey("mlh_head.fc.weight") && layerMLHHead != null)
        {
          layerMLHHead.LoadWeights(paramsToLoad, loadedParams, "mlh_head");
        }

        // 6) UNC heads
        layerUNCHead.LoadWeights(paramsToLoad, loadedParams, "unc_head");
        layerUncPolicyHead.LoadWeights(paramsToLoad, loadedParams, "unc_policy");

        // 7) Q deviation heads
        layerQDeviationLowerHead.LoadWeights(paramsToLoad, loadedParams, "qdev_lower");
        layerQDeviationUpperHead.LoadWeights(paramsToLoad, loadedParams, "qdev_upper");

        // 8) Smolgen layer
        if (TransformerConfig.SmolgenDimPerSquare > 0)
        {
          LinearLoad(paramsToLoad, loadedParams, smLinearShared, "smolgenPrepLayer.weight", "smolgenPrepLayer.bias");
        }

        // 9) Policy head
        layerPolicyHead.LoadWeights(paramsToLoad, loadedParams, "policy_head");

        // 10) Value heads
        (layerValueHead as NetTransformerLayerHead).LoadWeights(paramsToLoad, loadedParams, "value_head");
        (layerValue2Head as NetTransformerLayerHead).LoadWeights(paramsToLoad, loadedParams, "value2_head");

        // -- Now verify if any dictionary parameters were left unused
        bool foundMissing = false;
        foreach (KeyValuePair<string, Tensor> paramDefined in paramsToLoad)
        {
          if (!loadedParams.Contains(paramDefined.Key))
          {
            foundMissing = true;
            Console.WriteLine("Parameter not loaded: " + paramDefined);
          }
        }

        if (foundMissing)
        {
          Console.WriteLine(paramsToLoad.Count + " parameters defined");
          Console.WriteLine(loadedParams.Count + " parameters loaded");
          ConsoleUtils.WriteLineColored(ConsoleColor.Red,
             "SERIOUS WARNING: parameters not loaded, indicating "
             + "C# inference code is not up to date wrt. training code.");
          Console.WriteLine();
        }
      }
    }



    /// <summary>
    /// Prepare for LoRA training by freezing all parameters except those with "lora" in their name.
    /// </summary>
    void PrepareForLoRATraining()
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Wrapping for LoRA training (all other parameters frozen");

      TorchSharpUtils.FreezeLayers(this, l=>true, false);

      long numNotFrozenParams = 0;
      long nunNotFrozenLayers = 0;
      foreach ((string name, Parameter param) in this.named_parameters())
      {
        if (name.ToUpper().Contains("LORA"))
        {
          param.requires_grad = true;
          numNotFrozenParams += param.numel();
          nunNotFrozenLayers++;
        }        
      }

      Console.WriteLine("Total number of non-frozen layers/parameters:  " + nunNotFrozenLayers + "/" +  numNotFrozenParams);
      Console.WriteLine();

      LoRAEnabled = true;
    }


    /// <summary>
    /// Loads embedding weights from specified Dictionary of weights.
    /// </summary>
    /// <param name="weightsToLoadNamedParams"></param>
    public void LoadEmbeddingWeights(Dictionary<string, Tensor> weightsToLoadNamedParams)
    {
      HashSet<string> loadedParams = new();

      // Load weights, if provided.
      if (weightsToLoadNamedParams != null)
      {
        layerEmbedding.LoadWeights(weightsToLoadNamedParams, loadedParams);
      }
    }


    public Tensor lastOutputTrunk;


    public override (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) forward((Tensor squares, Tensor priorState) input)
    {
#if NOT
      Tensor flow;

      const int QBLUNDER_SLICE_BEGIN = 119;
      const int QBLUNDER_SLICE_END = 121;
      qblunders_negative_positive = squares[:, 0, QBLUNDER_SLICE_BEGIN: QBLUNDER_SLICE_END].clone().view(-1, 2)

    // insert zeros at the two qblunder slots in the main flow (masked out)
    flow = torch.cat((flow[:, :, :QBLUNDER_SLICE_BEGIN], torch.zeros_like(flow[:, :, QBLUNDER_SLICE_BEGIN: QBLUNDER_SLICE_END]), flow[:, :, QBLUNDER_SLICE_END:]), dim = 2)
#endif
      const int QBLUNDER_SLICE_BEGIN = 119;
      const int QBLUNDER_SLICE_END = 121;

      TensorIndex QBLUNDER_SLICE_LEFT = TensorIndex.Slice(null, QBLUNDER_SLICE_BEGIN);
      TensorIndex QBLUNDER_SLICE = TensorIndex.Slice(QBLUNDER_SLICE_BEGIN, QBLUNDER_SLICE_END);
      TensorIndex QBLUNDER_SLICE_RIGHT = TensorIndex.Slice(QBLUNDER_SLICE_END, null);

      // Equivalent to: squares[:, 0, 119:121].clone().view(-1, 2)
      Tensor qblunders_negative_positive = input.squares.index(new TensorIndex[] { TensorIndex.Colon, 0, QBLUNDER_SLICE }).clone().view(-1, 2);

      // Insert zeros at the two qblunder slots in the main flow
      // flow = torch.cat( (flow[:,:,:119], zeros_like(...), flow[:,:,121:]), dim=2 )
      Tensor flow = torch.cat([
            input.squares.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Colon, QBLUNDER_SLICE_LEFT }),
            zeros_like( input.squares.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Colon, QBLUNDER_SLICE })),
            input.squares.index(new TensorIndex[] { TensorIndex.Colon, TensorIndex.Colon, QBLUNDER_SLICE_RIGHT})
        ], 2);

      Tensor flowCS = layerEmbedding.call(flow);


      Tensor flowState = input.priorState;

      for (int layerNum = 0; layerNum < TransformerConfig.NumLayers; layerNum++)
      {
        using (NewDisposeScope())
        {
          // Run encoder.
          Tensor flowCSNext = layersEncodersArray[layerNum].call(flowCS, flowState);
          flowCS.Dispose();
          flowCS = flowCSNext.MoveToOuterDisposeScope();
        }
      }


      flowCS = headPremap.call(flowCS);
      flowCS = flowCS.reshape([-1, 64 * TransformerConfig.ModelDim/ HEAD_PREMAP_DIVISOR]);

//      flowCS = DebugCompareNetworkOutputs(flowCS, headSharedLinear.call(flowCS));
      flowCS = headSharedLinear.call(flowCS);

      Tensor flowValueHead = layerValueHead.call(flowCS);
      Tensor value2_input = torch.cat([flowCS, qblunders_negative_positive], -1);
      Tensor flowValue2Head = layerValue2Head.call(value2_input);

      Tensor flowPolicyHead = layerPolicyHead.call(flowCS);
      Tensor flowMLHHead = layerMLHHead == null ? null : layerMLHHead.call(flowCS);
      Tensor flowUNCHead = layerUNCHead.call(flowCS);
      Tensor flowUNCPolicyHead = layerUncPolicyHead.call(flowCS);

      Tensor flowQDeviationLowerHead = layerQDeviationLowerHead.call(flowCS);
      Tensor flowQDeviationUpperHead = layerQDeviationUpperHead.call(flowCS);
      Tensor flowAction = default;
      Tensor flowBoardState = default;
      flowCS.Dispose();

      // TODO: implement policy uncertainty, action uncertainty
      return (flowPolicyHead, flowValueHead, flowMLHHead, flowUNCHead, flowValue2Head, 
              flowQDeviationLowerHead, flowQDeviationUpperHead, flowUNCPolicyHead,
              flowAction, flowBoardState, default);
    }


    /// <summary>
    /// Compare specified output of a layer corresponding to the compareNetworkLayerName,
    /// by invoking the compareNetwork layer of same name and comparing 
    /// to the output of the specified output from this class.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="output"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    internal Tensor DebugCompareNetworkOutputs(Tensor input, Tensor output)
    {
      if (DebugComparisonNetwork == null)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, "NOTE NetTransformer.DebugComparisonNetwork is null, not dumping");
        return output;
      } 

      float[] compareOutput = default;
      foreach ((string name, Module module) layer in DebugComparisonNetwork.named_modules())
      {
        if (layer.name == DebugCompareLayerName)
        {
          Tensor tsOutput;

          if (layer.module is ScriptModule)
          {
            tsOutput = (Tensor)(layer.module as ScriptModule).to("cuda:0").to(ScalarType.Float16).call(input);
          }
          else
          {
            tsOutput = (layer.module as Module<Tensor, Tensor>).call(input);
          }
          compareOutput = tsOutput.type(ScalarType.Float32).cpu().data<float>().ToArray();
        }
      }
      if (compareOutput == null)
      {
        foreach ((string name, Module module) layer in DebugComparisonNetwork.named_modules())
        {
          Console.WriteLine(layer.name);
        }
        throw new Exception("no matching ScriptModule layer found from above for: " + DebugCompareLayerName);
      }

      float[] thisOutput = output.type(ScalarType.Float32).cpu().data<float>().ToArray();
      if (thisOutput.Length != compareOutput.Length)
      {
        throw new Exception("output length mismatch on " + DebugCompareLayerName);
      }
      Console.WriteLine("\r\nNetTransformer : compare values");
      for (int i = 0; i < thisOutput.Length; i++)
      {
        float absDiff = Math.Abs(compareOutput[i] - thisOutput[i]); 
        const float MAX_TOLERANCE = 0.002f;
        if (absDiff > MAX_TOLERANCE)
        {
          Console.WriteLine($"OUTPUT DIFFERENCE[{i}] of {absDiff} with values {compareOutput[i]} vs {thisOutput[i]}"); 
        } 
      }
      Console.WriteLine($"NetTransformer : compare values success on {DebugCompareLayerName}.");  
      return output;
    }


    public override void SetType(ScalarType type)
    {
      this.to(type);
    }


    public override (Tensor policy, Tensor value, Tensor mlh, Tensor unc,
                     Tensor value2, Tensor qDeviationLower, Tensor qDeviationUpper, Tensor policyUncertainty,
                     Tensor action, Tensor boardState, Tensor actionUncertainty,

                    FP16[] extraStats0, FP16[] extraStats1) Forward((Tensor squares, Tensor priorState)input)
    {
      (Tensor p, Tensor v, Tensor m, Tensor u, Tensor v2, Tensor qL, Tensor qU, Tensor pU, Tensor bs, Tensor a, Tensor aU) = forward((input.squares.to(ExecutionConfig.DataType), input.priorState));

      return (p, v, m, u, v2, qL, qU, pU, bs, a, aU, null, null);
    }
  }
}
