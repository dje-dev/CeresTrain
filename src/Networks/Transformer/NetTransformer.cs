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

using CeresTrain.NNIntrospection;
using CeresTrain.Utils;
using CeresTrain.TPG;
using CeresTrain.Trainer;


#endregion

namespace CeresTrain.Networks.Transformer
{
  /// <summary>
  /// Full Ceres Transformer model (embedding, encoder layer, heads).
  /// </summary>
  public class NetTransformer : CeresNeuralNet
  {
    const bool ALT_UP = false; // experimental, not found better so far.
    public readonly int AltUpK = 2;

    public const float MLH_DIVISOR = 100;

    public static bool DUMP_INFO_TO_CONSOLE = false;

    public NetTransformerDef TransformerConfig;
    public ConfigNetExecution ExecutionConfig;

    public ParameterStats ParameterStats => new(this);


    // First layer of head projects size down from embedding dimension by some factor
    // (after which all the independent squares have their outputs concatenated together.
    const int HEAD_PREMAP_DIVISOR_POLICY = 8;
    const int HEAD_PREMAP_DIVISOR_VALUE = 8;
    const int HEAD_PREMAP_DIVISOR_MLH = 16;
    const int HEAD_PREMAP_DIVISOR_UNC = 16;

    int FINAL_POLICY_FC1_SIZE => 128 * TransformerConfig.HeadWidthMultiplier;
    int FINAL_POLICY_FC2_SIZE => 64 * TransformerConfig.HeadWidthMultiplier;

    int FINAL_VALUE_FC1_SIZE => 32 * TransformerConfig.HeadWidthMultiplier;
    int FINAL_VALUE_FC2_SIZE => 8 * TransformerConfig.HeadWidthMultiplier;

    int FINAL_MLH_FC1_SIZE => 32 * TransformerConfig.HeadWidthMultiplier;
    int FINAL_MLH_FC2_SIZE => 8 * TransformerConfig.HeadWidthMultiplier;

    int FINAL_SUPPLEMENTAL_HEAD_FC1_SIZE => 32 * TransformerConfig.HeadWidthMultiplier;
    int FINAL_SUPPLEMENTAL_HEAD_FC2_SIZE => 8 * TransformerConfig.HeadWidthMultiplier;

    internal NetTransformerLayerEmbedding layerEmbedding;

    internal NetTransformerLayerEncoder[] layersEncodersArray;
    internal AltUpLayer[] altUpLayers;

    internal NetTransformerLayerHead layerValueHead;
    internal NetTransformerLayerHead layerValue2Head;
    internal NetTransformerLayerHead layerPolicyHead;
    internal NetTransformerLayerHead layerMLHHead;
    internal NetTransformerLayerHead layerUNCHead;
    internal NetTransformerLayerHead layerQDeviationLowerHead;
    internal NetTransformerLayerHead layerQDeviationUpperHead;

    Linear layerHeadReduce;

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
                          Dictionary<string, Tensor> overrideWeights = null) : base("CeresTransformer")
    {
      TransformerConfig = transformerNetDef;
      ExecutionConfig = executionConfig;

      float alpha = transformerNetDef.DeepNorm ? MathF.Pow(2 * TransformerConfig.NumLayers, 0.25f) : 1;

      if (DUMP_INFO_TO_CONSOLE)
      {
        Console.WriteLine("LOADING TORCHSCRIPT MODEL  " + executionConfig.SaveNetwork1FileName + " " + executionConfig.SaveNetwork2FileName);
      }

      using (torch.no_grad())
      {
        Dictionary<string, Tensor> paramsToLoad = null;

        // Potentially load weights (do here if Torchscript file, otherwise if C# dat file, do at end of constructor).
        if (overrideWeights != null)
        {
          paramsToLoad = new();
        }
        else if (ExecutionConfig.SaveNetwork1FileName != null &&
                (!ExecutionConfig.SaveNetwork1FileName.ToLower().EndsWith(".ts") &&
                 !ExecutionConfig.SaveNetwork1FileName.ToLower().EndsWith(".dat")))
        {
          throw new Exception("Specified ExecutionConfig.SaveNetwork1FileName must end with either .ts or .dat, instead see: " + ExecutionConfig.SaveNetwork1FileName);
        }
        else if (ExecutionConfig.SaveNetwork1FileName != null && !File.Exists(ExecutionConfig.SaveNetwork1FileName))
        {
          throw new Exception("Specified ExecutionConfig.SaveNetwork1FileName does not exist: " + ExecutionConfig.SaveNetwork1FileName);
        }
        else if (ExecutionConfig.SaveNetwork1FileName != null &&
                 ExecutionConfig.SaveNetwork1FileName.ToLower().EndsWith(".ts"))
        {
          ScriptModule<Tensor, (Tensor, Tensor, Tensor, Tensor)> transformerTS
            = TorchscriptUtils.TorchScriptFilesAveraged<Tensor, (Tensor, Tensor, Tensor, Tensor)>(ExecutionConfig.SaveNetwork1FileName,
                                                                                                    ExecutionConfig.SaveNetwork2FileName,
                                                                                                    ExecutionConfig.Device, ExecutionConfig.DataType);
          paramsToLoad = new();
          transformerTS.named_parameters().ToList().ForEach(p => paramsToLoad.Add(p.name, p.parameter));
        }

        HashSet<string> loadedParams = new();

        // EMBEDDING
        int embeddingWidth = TransformerConfig.ModelDim * (ALT_UP ? AltUpK : 1);
        layerEmbedding = new NetTransformerLayerEmbedding("embedding", TPGRecord.BYTES_PER_SQUARE_RECORD, embeddingWidth);
        layerEmbedding.to(ExecutionConfig.DataType).to(ExecutionConfig.Device);

        // Load weights, if provided.
        if (paramsToLoad != null)
        {
          layerEmbedding.LoadWeights(paramsToLoad, loadedParams);
        }

        if (TransformerConfig.SmolgenDimPerSquare > 0)
        {
          smLinearShared = Linear(TransformerConfig.SmolgenDim, 64 * 64, true, ExecutionConfig.Device, ExecutionConfig.DataType);
        }

        // ENCODER LAYERS
        layersEncodersArray = new NetTransformerLayerEncoder[TransformerConfig.NumLayers];
        for (int layerNum = 0; layerNum < TransformerConfig.NumLayers; layerNum++)
        {
          NetTransformerLayerEncoder teCS = new(TransformerConfig.NumLayers, layerNum, TransformerConfig.ModelDim,
                                                TransformerConfig.NumHeads, TransformerConfig.PreNorm,
                                                TransformerConfig.NormType, TransformerConfig.AttentionMultiplier,
                                                TransformerConfig.FFNMultiplier, TransformerConfig.FFNActivationType,
                                                alpha, ExecutionConfig.DropoutRate, ExecutionConfig.DropoutDuringInference,
                                                0, ExecutionConfig.SupplementaryStat == NNLayerMonitor.SupplementaryStatType.AverageCosineSimilarity,
                                                TransformerConfig.SmolgenDimPerSquare, TransformerConfig.SmolgenDim,
                                                TransformerConfig.SoftMoEConfig, ExecutionConfig.MonitorActivationStats, ref smLinearShared);
          teCS = teCS.to(ExecutionConfig.DataType).to(ExecutionConfig.Device);
          if (paramsToLoad != null)
          {
            teCS.LoadWeights(paramsToLoad, loadedParams);
          }
          layersEncodersArray[layerNum] = teCS;
          register_module($"encoder_layer_{layerNum}", teCS);
        }

        const bool SAVE_INTERMEDIATE_FOR_MLH_HEAD = false;

        layerMLHHead = new NetTransformerLayerHead("head_mlh", 64, TransformerConfig.ModelDim, HEAD_PREMAP_DIVISOR_MLH, FINAL_MLH_FC1_SIZE, FINAL_MLH_FC2_SIZE, 1,
                                                   TransformerConfig.HeadsActivationType, "RELU", SAVE_INTERMEDIATE_FOR_MLH_HEAD, false);
        layerMLHHead.to(ExecutionConfig.DataType).to(ExecutionConfig.Device);
        if (paramsToLoad != null)
        {
          layerMLHHead.LoadWeights(paramsToLoad, loadedParams, "mlhHeadPremap", "out_mlh_layer"); // reuse valueHeadPremap for MLH
        }

        layerUNCHead = new NetTransformerLayerHead("head_unc", 64, TransformerConfig.ModelDim, HEAD_PREMAP_DIVISOR_UNC, FINAL_SUPPLEMENTAL_HEAD_FC1_SIZE, FINAL_SUPPLEMENTAL_HEAD_FC2_SIZE, 1,
                                                   TransformerConfig.HeadsActivationType, "RELU", false, false);
        layerUNCHead.to(ExecutionConfig.DataType).to(ExecutionConfig.Device);
        if (paramsToLoad != null)
        {
          layerUNCHead.LoadWeights(paramsToLoad, loadedParams, "uncHeadPremap", "out_unc_layer"); // reuse valueHeadPremap for MLH
        }

        
        layerQDeviationLowerHead = new NetTransformerLayerHead("head_qdeviation_lower", 64, TransformerConfig.ModelDim, HEAD_PREMAP_DIVISOR_UNC, FINAL_SUPPLEMENTAL_HEAD_FC1_SIZE, FINAL_SUPPLEMENTAL_HEAD_FC2_SIZE, 1,
                                           TransformerConfig.HeadsActivationType, "RELU", false, false);
        layerQDeviationLowerHead.to(ExecutionConfig.DataType).to(ExecutionConfig.Device);
        if (paramsToLoad != null)
        {
          layerQDeviationLowerHead.LoadWeights(paramsToLoad, loadedParams, "qDevLowerHeadPremap", "out_qdev_lower_layer"); 
        }

        layerQDeviationUpperHead = new NetTransformerLayerHead("head_qdeviation_upper", 64, TransformerConfig.ModelDim, HEAD_PREMAP_DIVISOR_UNC, FINAL_SUPPLEMENTAL_HEAD_FC1_SIZE, FINAL_SUPPLEMENTAL_HEAD_FC2_SIZE, 1,
                                   TransformerConfig.HeadsActivationType, "RELU", false, false);
        layerQDeviationUpperHead.to(ExecutionConfig.DataType).to(ExecutionConfig.Device);
        if (paramsToLoad != null)
        {
          layerQDeviationUpperHead.LoadWeights(paramsToLoad, loadedParams, "qDevUpperHeadPremap", "out_qdev_upper_layer");
        }


        if (paramsToLoad != null && TransformerConfig.SmolgenDimPerSquare > 0)
        {
          ModuleParamLoadingUtils.LinearLoad(paramsToLoad, loadedParams, smLinearShared, "smolgenPrepLayer.weight", "smolgenPrepLayer.bias");
        }

        // POLICY HEAD
        layerPolicyHead = new NetTransformerLayerHead("head_policy", 64, TransformerConfig.ModelDim, HEAD_PREMAP_DIVISOR_POLICY, FINAL_POLICY_FC1_SIZE, FINAL_POLICY_FC2_SIZE, 1858,
                                                       TransformerConfig.HeadsActivationType, null, false, false);
        layerPolicyHead.to(ExecutionConfig.DataType).to(ExecutionConfig.Device);
        if (paramsToLoad != null)
        {
          layerPolicyHead.LoadWeights(paramsToLoad, loadedParams, "policyHeadPremap", "fcPolicyFinal");
        }

        // VALUE HEAD
        const bool SAVE_INTERMEDIATE_FOR_VALUE_HEAD = false;

        layerValueHead = new NetTransformerLayerHead("head_value", 64, TransformerConfig.ModelDim, HEAD_PREMAP_DIVISOR_VALUE,
                                                     FINAL_VALUE_FC1_SIZE, FINAL_VALUE_FC2_SIZE, 3,
                                                     TransformerConfig.HeadsActivationType, null, SAVE_INTERMEDIATE_FOR_VALUE_HEAD, false);
        layerValueHead.to(ExecutionConfig.DataType).to(ExecutionConfig.Device);
        if (paramsToLoad != null)
        {
          layerValueHead.LoadWeights(paramsToLoad, loadedParams, "valueHeadPremap", "out_value_layer");
        }

        layerValue2Head = new NetTransformerLayerHead("head_value2", 64, TransformerConfig.ModelDim, HEAD_PREMAP_DIVISOR_VALUE,
                                             FINAL_VALUE_FC1_SIZE, FINAL_VALUE_FC2_SIZE, 3,
                                             TransformerConfig.HeadsActivationType, null, SAVE_INTERMEDIATE_FOR_VALUE_HEAD, false);
        layerValue2Head.to(ExecutionConfig.DataType).to(ExecutionConfig.Device);

        if (paramsToLoad != null)
        {
          layerValue2Head.LoadWeights(paramsToLoad, loadedParams, "value2HeadPremap", "out_value2_layer");
        }

        if (ALT_UP)
        {
          altUpLayers = new AltUpLayer[TransformerConfig.NumLayers];
          for (int layerNum = 0; layerNum < TransformerConfig.NumLayers; layerNum++)
          {
            altUpLayers[layerNum] = new AltUpLayer(AltUpK, layerNum % AltUpK);
          }
          layerHeadReduce = Linear(TransformerConfig.ModelDim * AltUpK, TransformerConfig.ModelDim, hasBias: true);
        }

        if (paramsToLoad != null)
        {
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
            ConsoleUtils.WriteLineColored(ConsoleColor.Red, "SERIOUS WARNING: parameters not loaded, indicating C# inference code is not up to date wrt. training code.");
            Console.WriteLine();
          }

        }
      }

      RegisterComponents();

      if (overrideWeights == null
        && ExecutionConfig.SaveNetwork1FileName != null
        && executionConfig.SaveNetwork1FileName.ToLower().EndsWith(".dat"))
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
          ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Error loading Torchscript model, " 
                                     + "possible mismatch of C# model definition and saved model: " + ExecutionConfig.SaveNetwork1FileName
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
            // Set up the hook to monitor the layer.  
            node.module.register_forward_hook((module, input, output) =>
            {
              IntrinsicDimensionalitiesLastBatch = FP16.ToFP16(IntrinsicDimensionality.TwoNN(output));
              return output;
            });
          }
        }
      }

      const int MONITOR_TO_DUMP_RATIO = 20; // for efficiency, only monitor 1 out of 20 batches
      int monitorSkipCount = Math.Max(1, ExecutionConfig.ActivationMonitorDumpSkipCount / MONITOR_TO_DUMP_RATIO);
      NNLayerMonitorSet monitor = ExecutionConfig.ActivationMonitorDumpSkipCount > 0
                                //                                ? new NNLayerMonitorSet(this, 100, 1000, layerName => true)
                                ? new NNLayerMonitorSet(this, monitorSkipCount, ExecutionConfig.ActivationMonitorDumpSkipCount, layerName => true)
                                : null;

      // Optionally, set up a monitor for a specific layer.
      if (ExecutionConfig.SupplementaryStat != NNLayerMonitor.SupplementaryStatType.None)
      {
        // TODO: Currently we limit to output of encoder layers, generalize this.
        for (int i = 0; i < TransformerConfig.NumLayers; i++)
        {
          monitor?.SetSupplementalLayerStat($"encoder_layer_{i}", ExecutionConfig.SupplementaryStat);
        }
      }
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


    public override (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) forward(Tensor inputBoardSquares)
    {
      Tensor flowCS = layerEmbedding.call(inputBoardSquares);


      if (ALT_UP)
      { 
        // TODO: Split out a new class AltUpLayerStack and encapsulate this logic there.
        flowCS = flowCS.reshape(-1, 64, AltUpK, TransformerConfig.ModelDim);
        for (int layerNum = 0; layerNum < TransformerConfig.NumLayers; layerNum++)
        {
          int jStar = altUpLayers[layerNum].j_star;
          using (NewDisposeScope())
          {
            Tensor thisSub = flowCS[TensorIndex.Colon, TensorIndex.Colon, jStar, TensorIndex.Colon];
            Tensor flowCSNext = layersEncodersArray[layerNum].call(thisSub);

            flowCSNext = altUpLayers[layerNum].forward(flowCS, flowCSNext);

            flowCS.Dispose();
            flowCS = flowCSNext.MoveToOuterDisposeScope();
          }
        }

        flowCS = flowCS.reshape(-1, 64, TransformerConfig.ModelDim * AltUpK);
        flowCS = layerHeadReduce.call(flowCS);
      }
      else
      {
        for (int layerNum = 0; layerNum < TransformerConfig.NumLayers; layerNum++)
        {
          using (NewDisposeScope())
          {
            Tensor flowCSNext = layersEncodersArray[layerNum].call(flowCS);

            flowCS.Dispose();
            flowCS = flowCSNext.MoveToOuterDisposeScope();
          }
        }

      }

      Tensor flowValueHead = layerValueHead.call(flowCS);
      Tensor flowValue2Head = layerValue2Head.call(flowCS);
      Tensor flowPolicyHead = layerPolicyHead.call(flowCS);
      Tensor flowMLHHead = layerMLHHead.call(flowCS);
      Tensor flowUNCHead = layerUNCHead.call(flowCS);

      Tensor flowQDeviationLowerHead = layerQDeviationLowerHead.call(flowCS);
      Tensor flowQDeviationUpperHead = layerQDeviationUpperHead.call(flowCS);

      flowCS.Dispose();

      return (flowPolicyHead, flowValueHead, flowMLHHead, flowUNCHead, flowValue2Head, flowQDeviationLowerHead, flowQDeviationUpperHead);
    }


    public override void SetType(ScalarType type)
    {
      this.to(type);
    }

    public override (Tensor value, Tensor policy, Tensor mlh, Tensor unc, 
                     Tensor value2, Tensor qDeviationLower, Tensor qDeviationUpper,
                    FP16[] extraStats0, FP16[] extraStats1) Forward(Tensor inputSquares, Tensor inputMoves)
    {
      (Tensor p, Tensor v, Tensor m, Tensor u, Tensor v2, Tensor qL, Tensor qU) = forward(inputSquares.to(ExecutionConfig.DataType));

      return (v, p, m, u, v2, qL, qU, null, null);
    }
  }
}
