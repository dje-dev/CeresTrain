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
using System.Diagnostics;
using System.IO;
using System.Linq;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using static TorchSharp.torch.optim.lr_scheduler.impl;
using static TorchSharp.torch.utils.data;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;

using Ceres.Chess.Positions;
using Ceres.Chess.NNEvaluators.Ceres;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators.Ceres.TPG;

using CeresTrain.NNEvaluators;
using CeresTrain.Utils;
using CeresTrain.Utils.Tensorboard;
using CeresTrain.TrainData.TPGDatasets;
using CeresTrain.TPGDatasets;
using CeresTrain.Networks;
using CeresTrain.Networks.MiscModules;
using CeresTrain.UserSettings;
using CeresTrain.TPG.TPGGenerator;
using CeresTrain.Networks.SoftMoE;

using CeresTrain.TrainCommands;
using CeresTrain.Networks.Transformer;
using CeresTrain.Optimizers;

#endregion

namespace CeresTrain.Trainer
{
  public enum TransformerTypeEnum
  {
    TorchscriptModule,
    PyTorchTransformer,
    CeresTransformer
  };



  public partial class CeresTrainCommandTrain : CeresTrainCommandBase, IDisposable
  {
    const TransformerTypeEnum TransformerType = TransformerTypeEnum.CeresTransformer;

    public long MaxPositions;

    public long NumParameters;

    TrainingStatusTable consoleStatusTable;
    TensorboardWriter tbWriter;

    Tensor mlhTarget;
    Tensor uncTarget;
    Tensor uncPolicyTarget;
    Tensor valueTarget;
    Tensor policyTarget;
    Tensor maskedPolicyForLoss;
    Tensor value2Target;
    Tensor qDeviationLowerTarget;
    Tensor qDeviationUpperTarget;

    Tensor lossValueBatch;
    Tensor lossPolicyBatch;
    Tensor lossMLHBatch;
    Tensor lossUNCBatch;

    Tensor lossValue2Batch;
    Tensor lossValueDBatch;
    Tensor lossValue2DBatch;
    Tensor lossAction2DBatch;
    Tensor lossQDevLowerBatch;
    Tensor lossQDevUpperBatch;
    Tensor lossUNCPolicyBatch;

    Tensor lossTotal;

    Optimizer optimizer;

    int batchId = 0;

    float[] predictionWinLoss;

    DateTime timeStartTraining = DateTime.Now;
    DateTime timeLastDump = DateTime.Now;
    DateTime timeLastSave = DateTime.Now;

    CeresTrainerMonitor monitor;



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="config"></param>
    /// <param name="description"></param>
    /// <param name="statusTable"></param>
    public CeresTrainCommandTrain(in ConfigTraining config, string description, TrainingStatusTable statusTable) : base(in config)
    {
      consoleStatusTable = statusTable;
    }
    

    /// <summary>
    /// Sets the underlying CeresNeuralNet to a specified value.
    /// </summary>
    /// <param name="model"></param>
    public void SetModel(CeresNeuralNet model)
    {
      Model = model;
    }


    /// <summary>
    /// Initializes the trainer before any commands performed.
    /// </summary>
    public void PrepareTrainer()
    {
      if (Model == null)
      {
        Model = TrainingConfig.CreateCeresNeuralNet();
      }
    }


    public static int WDLToMostProbableV(float vW, float vD, float vL)
    {
      float max = MathF.Max(vW, MathF.Max(vD, vL));
      return vW == max ? 1 : vD == max ? 0 : -1;
    }




    private void Train(string modelTag,
                       CeresNeuralNet model,
                       IModuleNNEvaluator anchorEvaluator,
                       Optimizer optimizer,
                       Loss<Tensor, Tensor, Tensor> lossPolicy,
                       Loss<Tensor, Tensor, Tensor> lossValue,
                       Loss<Tensor, Tensor, Tensor> lossValue2,
                       Loss<Tensor, Tensor, Tensor> lossMLH,
                       Loss<Tensor, Tensor, Tensor> lossUNC,
                       Loss<Tensor, Tensor, Tensor> lossUNCPolicy,
                       Loss<Tensor, Tensor, Tensor> lossQDeviationLower,
                       Loss<Tensor, Tensor, Tensor> lossQDeviationUpper,
                       Loss<Tensor, Tensor, Tensor> lossValueD,
                       Loss<Tensor, Tensor, Tensor> lossValue2D,
                       DataLoader dataLoader,
                       Dataset tpgDataset,
                       LRScheduler scheduler,
                       TensorboardWriter tbWriter,
                       long maxPositions,
                       ref long numRead,
                       bool silentMode = false)
    {
      MaxPositions = maxPositions;
      predictionWinLoss = new float[OptimizationBatchSizeForward];
      numRead = 0;

      sumFineTuneAnchorErrors = 0;
      sumFineTuneNonAnchorErrors = 0;
      countFineTuneAnchorPositions = 0;
      countFineTuneNonAnchorPositions = 0;

      int batchSizeData = TrainingConfig.OptConfig.BatchSizeForwardPass;
      int batchSizeOptimization = TrainingConfig.OptConfig.BatchSizeBackwardPass;
      if (batchSizeOptimization < batchSizeData
       || batchSizeOptimization % batchSizeData != 0)
      {
        throw new Exception("Invalid TrainingDataConfig.BatchSize/TrainingConfig.OptimizationConfig.BatchSize combination "
                            + $"{batchSizeData}/{batchSizeOptimization}");
      }
      int numBatchesGradientAccumulate = batchSizeOptimization / batchSizeData;

      monitor = new(this);

      int batchAccumulationCounter = 0;// counter to keep track of batches

      using (var d = NewDisposeScope())
      {
        foreach (Dictionary<string, Tensor> batch in dataLoader)
        {
          numRead += OptimizationBatchSizeForward;

          model.train();

          Tensor inputSquares = batch["squares"].reshape(OptimizationBatchSizeForward, TPGRecord.NUM_SQUARES, TPGRecord.BYTES_PER_SQUARE_RECORD);
          mlhTarget = batch.ContainsKey("mlh") ? batch["mlh"].reshape(OptimizationBatchSizeForward, 1) : default;
          uncTarget = batch["unc"].reshape(OptimizationBatchSizeForward, 1);
          uncPolicyTarget = batch["unc_policy"].reshape(OptimizationBatchSizeForward, 1);
          valueTarget = batch["wdl"].reshape(OptimizationBatchSizeForward, 3);
          policyTarget = batch["policy"].reshape(OptimizationBatchSizeForward, 1858);

          value2Target = batch["wdl2"].reshape(OptimizationBatchSizeForward, 3);
          qDeviationLowerTarget = batch["q_deviation_lower"].reshape(OptimizationBatchSizeForward, 1);
          qDeviationUpperTarget = batch["q_deviation_upper"].reshape(OptimizationBatchSizeForward, 1);
          bool isAnchorBatch = batch.ContainsKey("is_set1");
//Console.WriteLine(isAnchorBatch + " " + value2Target.shape[0]);
          const bool RUN_INFERENCE_BENCHMARK = false;
          if (RUN_INFERENCE_BENCHMARK)
          {
            const bool INFERENCE_FP16 = true;
            var TYPE = INFERENCE_FP16 ? ScalarType.Float16 : ScalarType.Float32;

            model = model.to(TYPE);
            inputSquares = inputSquares.to(TYPE);
            inputSquares.requires_grad = false;
            model.eval();
            while (true)
            {
              //              evaluator.forwardValuePolicy(NetConfig.ATTENTION ? inputMovesSquares : inputRawBoards, squareMasks);
              using (new TimingBlock("evals 100 times with batch size " + OptimizationBatchSizeForward))
              {
                for (int i = 0; i < 100; i++)
                {
                  using (no_grad())
                  {
                    using (var dx = NewDisposeScope())
                    {
                      throw new NotImplementedException();
                      //var result = evaluator.forwardValuePolicyMLH_UNC(inputSquares, inputMoves);
                      //var zz = result.policy.sum().to(CPU);
                    }
                  }
                }
              }
            }
          }

          Tensor value;
          Tensor policy;
          Tensor mlh;
          Tensor unc;
          Tensor policy1858;
          Tensor value2;
          Tensor qDeviationLower;
          Tensor qDeviationUpper;
          Tensor action;
          Tensor boardState;
          Tensor uncertaintyPolicy;
          Tensor actionUncertainty;

          FP16[] extraStats0;
          FP16[] extraStats1;
          if (model is IModuleNNEvaluator)
          {
            IModuleNNEvaluator evaluator = model as IModuleNNEvaluator;

            // already scaled inputSquares = inputSquares.to(TrainingConfig.ExecConfig.DataType).div(ByteScaled.SCALING_FACTOR);

            (policy, value, mlh, unc, value2, qDeviationLower, qDeviationUpper, uncertaintyPolicy,
             action, boardState, actionUncertainty, extraStats0, extraStats1) = evaluator.forwardValuePolicyMLH_UNC((inputSquares, null));
          }
          else
          {
            throw new NotImplementedException("next line needs remediation?");
#if NOT
            var tsModel = (jit.ScriptModule<Tensor, (Tensor, Tensor, Tensor, Tensor)>)model;
            var tsModelOutput = tsModel.call(inputSquares);
            value = tsModelOutput.Item2;
            policy1858 = policy = tsModelOutput.Item1;
            mlh = default;
            unc = default;
#endif
          }

          bool USE_4BOARD = (object)action != null;

          // Mask illegal moves out of the policy vector.
          const float MASK_POLICY_VALUE = -6E4f; // say within float16 range in case that is underlying data type
          Tensor legalMoves = policyTarget.greater(0);
          Tensor illegalMaskValue = zeros_like(policy).add_(MASK_POLICY_VALUE);
          maskedPolicyForLoss = where(legalMoves, policy, illegalMaskValue);

          const float QDEV_LOSS_POST_SCALE = 10.0f;
          const float MLH_LOSS_POST_SCALE = 5.0f;
          const float UNC_POLICY_LOSS_POST_SCALE = 10.0f;
          const float UNC_VALUE_LOSS_POST_SCALE = 150.0f;

          // .......................................................
          // TODO: have better way of identifying which to use anchor on than the modulus check here ******
          bool IS_ANCHOR = anchorEvaluator != null && isAnchorBatch;
          bool IS_FINE_TUNE = anchorEvaluator != null && !isAnchorBatch;
          if (IS_ANCHOR)
          {
            (Tensor policy, Tensor value, Tensor mlh, Tensor unc,
             Tensor value2, Tensor qDeviationLower, Tensor qDeviationUpper,
             Tensor uncertaintyPolicy, Tensor action, Tensor boardState, Tensor actionUncertainty,
             FP16[] extraStats0, FP16[] extraStats1) anchorOutput = default;

            using (torch.no_grad())
            {
              (anchorEvaluator as NetTransformer).LoRAEnabled = false;
              anchorOutput = anchorEvaluator.forwardValuePolicyMLH_UNC((inputSquares, null));
              (anchorEvaluator as NetTransformer).LoRAEnabled = true;
            }

            valueTarget = anchorOutput.value;
            policyTarget = where(legalMoves, anchorOutput.policy, illegalMaskValue); 
            mlhTarget = anchorOutput.mlh;
            uncTarget = anchorOutput.unc;
            uncPolicyTarget = anchorOutput.uncertaintyPolicy;
            value2Target = anchorOutput.value2;
            qDeviationLowerTarget = anchorOutput.qDeviationLower;
            qDeviationUpperTarget = anchorOutput.qDeviationUpper;
          }
          // .......................................................


          // TODO: Someday short circuit evaluation if weight is zero.
          if (IS_ANCHOR)
          {
            // Convert from logits to probabilities as expected by CrossEntropyLoss.
            policyTarget = softmax(policyTarget, dim: -1);
            valueTarget = softmax(valueTarget, dim: -1);
            value2Target = softmax(value2Target, dim: -1);
          }
          
          lossPolicyBatch = lossPolicy.call(maskedPolicyForLoss, policyTarget);
          lossValueBatch = lossValue.call(value, valueTarget);
          lossValue2Batch = lossValue2.call(value2, value2Target);

          // Subtract entropy from some of the losses
          lossValueBatch = lossValueBatch - TorchSharpUtils.Entropy(valueTarget);
          lossValue2Batch = lossValue2Batch - TorchSharpUtils.Entropy(value2Target);
          lossPolicyBatch = lossPolicyBatch - TorchSharpUtils.Entropy(policyTarget);
          

          lossMLHBatch = (object)mlh == null ? zeros_like(unc) : lossMLH.call(mlh, mlhTarget) * MLH_LOSS_POST_SCALE;
          lossUNCBatch = lossUNC.call(unc, uncTarget) * UNC_VALUE_LOSS_POST_SCALE;
          lossUNCPolicyBatch = lossUNCPolicy.call(uncertaintyPolicy, uncPolicyTarget) * UNC_POLICY_LOSS_POST_SCALE;

          // lossActionBatch = lossActionBatch - TorchSharpUtils.Entropy(actionTarget);
          // lossValueDBatch = lossQDeviationLower.call(qDeviationLower, qDeviationLowerTarget) * QDEV_LOSS_SCALE;
          // lossValue2DBatch = lossQDeviationUpper.call(qDeviationUpper, qDeviationUpperTarget) * QDEV_LOSS_SCALE;

          lossQDevLowerBatch = lossQDeviationLower.call(qDeviationLower, qDeviationLowerTarget) * QDEV_LOSS_POST_SCALE;
          lossQDevUpperBatch = lossQDeviationUpper.call(qDeviationUpper, qDeviationUpperTarget) * QDEV_LOSS_POST_SCALE;

          if (IS_FINE_TUNE)
          {
            // In fine tune mode only Value1 and Policy are
            // considered reliable as fine-tuning targets.
            lossValue2Batch = zeros_like(lossValue2Batch);
            lossMLHBatch = zeros_like(lossMLHBatch);
            lossUNCBatch = zeros_like(lossUNCBatch);
            lossUNCPolicyBatch = zeros_like(lossUNCPolicyBatch);
            lossQDevLowerBatch = zeros_like(lossQDevLowerBatch);
            lossQDevUpperBatch = zeros_like(lossQDevUpperBatch);
          }

          // lossAction2DBatch = lossAction.call()
          lossTotal = 
                      lossPolicyBatch * TrainingConfig.OptConfig.LossPolicyMultiplier
                    + lossValueBatch * TrainingConfig.OptConfig.LossValueMultiplier
                    + lossValue2Batch * TrainingConfig.OptConfig.LossValue2Multiplier
                    + lossUNCBatch * TrainingConfig.OptConfig.LossUNCMultiplier
                    + lossUNCPolicyBatch * TrainingConfig.OptConfig.LossUncertaintyPolicyMultiplier
                    + lossQDevLowerBatch * TrainingConfig.OptConfig.LossQDeviationMultiplier
                    + lossQDevUpperBatch * TrainingConfig.OptConfig.LossQDeviationMultiplier;

          if (float.IsNaN(lossTotal.to(ScalarType.Float32).item<float>()))
          {
            throw new Exception("lossTotal was NaN");
          }

          if (IS_ANCHOR)
          {
            // TODO: check policy error sometimes negative (?)            
            if (false)
            {
              Console.WriteLine();
              Console.WriteLine("policy " + MathF.Round(lossPolicyBatch.cpu().to(ScalarType.Float32).item<float>() * TrainingConfig.OptConfig.LossPolicyMultiplier, 3));
              Console.WriteLine("value " + MathF.Round(lossValueBatch.cpu().to(ScalarType.Float32).item<float>() * TrainingConfig.OptConfig.LossValueMultiplier, 3));
              Console.WriteLine("value2 " + MathF.Round(lossValue2Batch.cpu().to(ScalarType.Float32).item<float>() * TrainingConfig.OptConfig.LossValue2Multiplier, 3));
              Console.WriteLine("unc " + MathF.Round(lossUNCBatch.cpu().to(ScalarType.Float32).item<float>() * TrainingConfig.OptConfig.LossUNCMultiplier, 3));
              Console.WriteLine("uncPolicy " + MathF.Round(lossUNCPolicyBatch.cpu().to(ScalarType.Float32).item<float>() * TrainingConfig.OptConfig.LossUncertaintyPolicyMultiplier, 3));
              Console.WriteLine("qDevLower " + MathF.Round(lossQDevLowerBatch.cpu().to(ScalarType.Float32).item<float>() * TrainingConfig.OptConfig.LossQDeviationMultiplier, 3));
              Console.WriteLine("qDevUpper " + MathF.Round(lossQDevUpperBatch.cpu().to(ScalarType.Float32).item<float>() * TrainingConfig.OptConfig.LossQDeviationMultiplier, 3));
            }

            sumFineTuneAnchorErrors += lossTotal.to(ScalarType.Float32).item<float>();
            countFineTuneAnchorPositions += (int)inputSquares.shape[0];
          }
          else if (IS_FINE_TUNE)
          {
            sumFineTuneNonAnchorErrors += lossTotal.to(ScalarType.Float32).item<float>();
            countFineTuneNonAnchorPositions += (int)inputSquares.shape[0];
          }

          if ((object)mlh is not null)
          {
            throw new NotImplementedException();
            //            +lossMLHBatch * TrainingConfig.OptConfig.LossMLHMultiplier

          }
          if (USE_4BOARD)
          {
            throw new Exception("Incomplete, need losses for others like lossValueDBatch, lossValue2DBatch");
            //            lossTotal = lossTotal + lossAction2DBatch * TrainingConfig.OptConfig.LossActionMultiplier;
          }

          lossTotal.backward();
          if ((batchAccumulationCounter + 1) % numBatchesGradientAccumulate == 0)
          {
            if (TrainingConfig.OptConfig.GradientClipLevel > 0)
            {
              nn.utils.clip_grad_norm_(parametersNotFrozen, TrainingConfig.OptConfig.GradientClipLevel);
            }

            scheduler?.step();
            optimizer.step();
            optimizer.zero_grad();
          }

          batchAccumulationCounter++;

          double secsSinceLastDump = (DateTime.Now - timeLastDump).TotalSeconds;
          double secsSinceStartTraining = (DateTime.Now - timeStartTraining).TotalSeconds;
          const float UPDATE_STATS_INTERVAL_SECS = 1;
          bool dumpStatsThisBatch = secsSinceLastDump > UPDATE_STATS_INTERVAL_SECS;

          if (numRead > MaxPositions)
          {
            if (!silentMode)
            {
              // End training.Dump stats one last time and save final network.
              DumpTrainingStatsToConsole("LOCAL", value, policy, silentMode, ref numRead);
              SaveNetwork(model, optimizer, false);
            }

            break;
          }

          if (TEST_EVAL_FROM_CSHARP)
          {
            Console.WriteLine();
            Console.WriteLine();
            DumpTestEvalStats();
          }

          if (dumpStatsThisBatch && !silentMode)
          {
            DumpTrainingStatsToConsole("LOCAL", value, policy, silentMode, ref numRead);
          }

          batchId++;
          d.DisposeEverything();
        }
      }
    }


    private DateTime PossiblySaveNetwork(CeresNeuralNet model, Optimizer optimizer,
                                         DateTime timeLastSave, bool silentMode, bool isLowestLossSoFar = false)
    {
      const int SAVE_INTERVAL_SECONDS = 60 * 10;

      double secondsSinceSave = (DateTime.Now - timeLastSave).TotalSeconds;
      if (!silentMode)
      {

        if (isLowestLossSoFar || secondsSinceSave > SAVE_INTERVAL_SECONDS)
        {
          timeLastSave = SaveNetwork(model, optimizer, isLowestLossSoFar);
        }
      }

      return timeLastSave;
    }


    string SaveNetFileName(bool best, bool optimizer) => Path.Combine(CeresTrainUserSettingsManager.Settings.OutputNetsDir,
                                                                      (optimizer ? "opt_" : "model_")
                                                                    + (timeStartTraining.Ticks % 10_000).ToString()
                                                                    + (best ? "_best.dat" : ".dat"));

    string saveNetworkFileName;

    private DateTime SaveNetwork(CeresNeuralNet model, Optimizer optimizer, bool isLowestLossSoFar)
    {
      string saveFNWeights = SaveNetFileName(isLowestLossSoFar, false);
      string saveFNOpt = SaveNetFileName(isLowestLossSoFar, true);

      model.save(saveFNWeights);
      if (TransformerType == TransformerTypeEnum.TorchscriptModule)
      {
        throw new NotImplementedException();
        //torch.jit.save((model as ModelEmbeddingMHA).transformerTS, saveFNOpt);
      }
      else
      {
        (optimizer as OptimizerHelper).save_state_dict(saveFNOpt);
      }

      timeLastSave = DateTime.Now;
      saveNetworkFileName = saveFNWeights;
      return timeLastSave;
    }



    public TrainingResultSummary DoTrain(string trainingSessionDescription, IModuleNNEvaluator anchorEvaluator = null)
    {
      PrepareTrainer();

      // TODO: need to load in original data type even if using another for inference
      //      model.to(device).to(ScalarType.BFloat16); 

      Model.to(TrainingConfig.ExecConfig.Device, TrainingConfig.ExecConfig.DataType);

      // Optionally train the PyTorch model directly
      if (false)
      {
        throw new Exception("Need remediation below?");
        //        var transformerTS = TorchscriptUtils.TorchscriptFilesAveraged<Tensor, (Tensor, Tensor, Tensor, Tensor)>(CeresNetworkTesting.TS_FN1a, CeresNetworkTesting.TS_FN1b, device, TrainingConfig.TransformerConfig.DataType);
        //        model = transformerTS;
        //        model = model.to(device, TrainingConfig.TransformerConfig.DataType);
      }

      if (TrainingConfig.OptConfig.CheckpointResumeFromFileName != null)
      {
        Model.load(TrainingConfig.OptConfig.CheckpointResumeFromFileName);
      }

      if (TEST_EVAL_FROM_CSHARP)
      {
        evaluatorCeres = new(NNEvaluatorInferenceEngineType.CSharpViaTorchscript, Model,
                             TrainingConfig.ExecConfig.Device, TrainingConfig.ExecConfig.DataType, true, false,
                             options: new NNEvaluatorOptionsCeres());
        Console.WriteLine("baseline eval: " + evaluatorCeres.Evaluate(Ceres.Chess.Position.StartPosition));
      }

      // Set up validation evaluator and buffer.
      // Create new evaluator from this model (note that never set to FP16).
      //Console.WriteLine("VALIDATION TESTS WILL BE OF SIZE " + NUM_VALIDATION_POS + " using " + TPGNetTests.VALIDATION_ZST_FN());
      //  DISABLED     validationTSEvaluator = new NNEvaluatorTorchsharp(model as ModelEmbeddingMHA, null, device.type, device.index, NetConfig.EmitPlySinceLastMovePerSquare, false);

      Dataset tpgDataset = BuildDataSet(TrainingConfig.DataConfig.TARPositionSkipCount);

      using (DataLoader dataLoader = new DataLoader(tpgDataset, 1, device: TrainingConfig.ExecConfig.Device))
      {
        bool SILENT_MODE = anchorEvaluator != null;
        TrainingLoop(TrainingConfig.ExecConfig.ID, Model, anchorEvaluator,
                     TrainingConfig.OptConfig.CheckpointResumeFromFileName,
                     0, // unknown StartingCheckpointLastPosNum,
                     dataLoader, tpgDataset, trainingSessionDescription, SILENT_MODE);
        // TODO: tpgDataset should be a  member of class, not passed explicitly
      }

      TrainingLossSummary lossSummary = new(thisLossAdjRunning, lossValueAdjRunning, valueAccAdjRunning,
                                            lossPolicyAdjRunning, policyAccAdjRunning,
                                            lossMLHAdjRunning, lossUNCAdjRunning,
                                            lossValue2AdjRunning,
                                            float.NaN, float.NaN, float.NaN,
                                            valueDLossAdjRunning, value2DLossAdjRunning,
                                            0, 0  /* TODO: actionLossRunning*/,
                                            countFineTuneAnchorPositions,
                                            countFineTuneNonAnchorPositions,
                                            sumFineTuneAnchorErrors /countFineTuneAnchorPositions, 
                                            sumFineTuneNonAnchorErrors/countFineTuneNonAnchorPositions);

      tpgDataset.Dispose();
      //      Dispose();

      return new TrainingResultSummary(Environment.MachineName, Environment.MachineName, TrainingConfig.ExecConfig.ID,
                                       DateTime.Now, "SUCCESS", NumParameters, DateTime.Now - timeStartTraining,
                                       numPositionsReadFromTraining, lossSummary, null,
                                       SaveNetFileName(false, false), SaveNetFileName(true, false));
    }


    public static IEnumerator<(TPGRecord[], bool)> EnumeratorAddTrue(IEnumerator<TPGRecord[]> source)
    {
      while (source.MoveNext())
      {
        yield return (source.Current, true);
      }
    }


    private Dataset BuildDataSet(int skipCount)
    {
//      Console.WriteLine("Generating training data from tablebase from  " + TrainingConfig.DataConfig.SourceType);

      Dataset tpgDataset;
      switch (TrainingConfig.DataConfig.SourceType)
      {
        case ConfigData.DataSourceType.DirectFromTPGFixedSet:
        case ConfigData.DataSourceType.PreprocessedFromTAR:
        case ConfigData.DataSourceType.DirectFromTPG:
          {

            // Using larger parallelism will:
            //   - improve shuffling of data seen by neural network while training
            //   - require more GPU memory
            int NUM_PARALLEL_DATASET = 4;

            IEnumerator<(TPGRecord[], bool)> enumerator = null;
            if (TrainingConfig.DataConfig.SourceType == ConfigData.DataSourceType.PreprocessedFromTAR)
            {
              TPGGeneratorOptions.DeblunderType DEBLUNDER_TYPE = TPGGeneratorOptions.DeblunderType.PositionQ;

              IEnumerator<TPGRecord[]> enumeratorBase = TPGTorchDatasetComboHelpers.GeneratorTPGRecordsViaGeneratorFromV6(TrainingConfig.DataConfig.TrainingFilesDirectory, null,
                                                                                             long.MaxValue, TrainingConfig.OptConfig.BatchSizeForwardPass,
                                                                                             DEBLUNDER_TYPE,
                                                                                             allowFilterOutRepeatedPositions: false,
                                                                                             rescoreTablebases: true,
                                                                                             partiallyFilterObviousDrawsAndWins: false,
                                                                                             verbose: true,
                                                                                             skipCount: skipCount);
              enumerator = EnumeratorAddTrue(enumeratorBase);
            }
            else if (TrainingConfig.DataConfig.SourceType == ConfigData.DataSourceType.DirectFromTPGFixedSet)
            {
              NUM_PARALLEL_DATASET = 1; // parallelism probably not properly supported by TPGRecordBatchProvider
              enumerator = new TPGRecordBatchProvider(TrainingConfig.DataConfig.TPGFixedSet1, TrainingConfig.DataConfig.TPGFixedSet2,
                                                      TrainingConfig.OptConfig.BatchSizeForwardPass,
                                                      TrainingConfig.DataConfig.NumTPGFixedSet1BatchesReturnedForSet2
                                                      ).GetEnumerator();
            }

            //TPGTorchDatasetCombo tpgDataset = new (TPGRecord.NUM_SQUARES, TPG_DIR, 0, 0, NetConfig.FractionQ, device, NetConfig.BATCH_SIZE, enumerator);
            tpgDataset = new TorchDatasetFromTPG(TrainingConfig.DataConfig.TrainingFilesDirectory, TrainingConfig.DataConfig.FractionQ,
                                                 TrainingConfig.ExecConfig.Device, TrainingConfig.ExecConfig.DataType,
                                                 TrainingConfig.OptConfig.BatchSizeForwardPass, TrainingConfig.DataConfig.WDLLabelSmoothing,
                                                 enumerator, NUM_PARALLEL_DATASET);
            break;
          }

        case ConfigData.DataSourceType.DirectFromPositionGenerator:
          const bool SUCCEED_IF_INCOMPLETE_DTZ_INFO = true;
          tpgDataset = new TorchDatasetFromPositionGenerator(TrainingConfig.DataConfig.PositionGenerator, SUCCEED_IF_INCOMPLETE_DTZ_INFO,
                                                             TrainingConfig.ExecConfig.DataType, TrainingConfig.ExecConfig.Device,
                                                             TrainingConfig.OptConfig.BatchSizeForwardPass,
                                                             TrainingConfig.DataConfig.WDLLabelSmoothing);
          break;
        default:
          throw new Exception("Unsupported TrainingConfig.Data.SourceType");
      }

      return tpgDataset;
    }


    public AdamW.Options AdamWOptionsWithWD(float wd)
    {
      return new AdamW.Options()
      {
        LearningRate = TrainingConfig.OptConfig.LearningRateBase,
        InitialLearningRate = TrainingConfig.OptConfig.LearningRateBase * TrainingConfig.OptConfig.LRWarmupPhaseMultiplier,
        weight_decay = wd,
        beta1 = TrainingConfig.OptConfig.Beta1,
        beta2 = TrainingConfig.OptConfig.Beta2        
      };
    }

    long numPositionsReadFromTraining = 0;


    Parameter[] parametersNotFrozen;
    internal void TrainingLoop(string tag, CeresNeuralNet model, IModuleNNEvaluator anchorEvaluator,
                               string startingCheckpointFN, long startingCheckpointLastNumPos,
                               DataLoader train, Dataset tpgDataset,
                               string trainingSessionDescription, bool silentMode)
    {
      long lastFlushNumPositions = 0;

      // Extract set of parameters once for reuse below.
      parametersNotFrozen = model.parameters().Where(p => p.requires_grad).ToArray();

      if (!silentMode)
      {
        // Log to tensorboard files.
        // Tensorboard can be used to view output, for example change to the output log directory and run:
        //   tensorboard--logdir =. --bind_all --port 6006
        string logDirName = CeresTrainUserSettingsManager.Settings.OutputLogsDir;
        tbWriter = new TensorboardWriter(logDirName, tag);
      }

      if (TrainingConfig.OptConfig.Optimizer == OptimizerType.SGD)
      {
        // TODO: Not yet implemented is per-parameter weight decay, etc.
        //       Settings (such as momentum) probably not synced up with PyTorch implementation.
        optimizer = SGD(parametersNotFrozen,
                        TrainingConfig.OptConfig.LearningRateBase,
                        //                        momentum: 0.9, 
                        //                        nesterov: true,
                        weight_decay: TrainingConfig.OptConfig.WeightDecay);
      }
      else if (TrainingConfig.OptConfig.Optimizer == OptimizerType.AdamW)
      {
        AdamW.ParamGroup[] adamWOptGroups = GetOptimizerAdamWGroups(model);
        optimizer = AdamW(adamWOptGroups);
      }
      else if (TrainingConfig.OptConfig.Optimizer == OptimizerType.Muon)
      {
        AdamW.ParamGroup[] adamWOptGroups = GetOptimizerAdamWGroups(model);

        List<Parameter> muonParams = model.named_parameters().Where(kv => kv.parameter.requires_grad && kv.parameter.ndim >= 2 && !kv.name.Contains("embedding")/* && kv.name.Contains("transformer_layer")*/).Select(kv => kv.parameter).ToList();
        List<Parameter> adamwParams = model.named_parameters().Where(kv => kv.parameter.requires_grad && (kv.parameter.ndim < 2 || kv.name.Contains("embedding")/* || !kv.name.Contains("transformer_layer")*/)).Select(kv => kv.parameter).ToList();

        optimizer = MuonOptimizerHelper.MuonOptimizer(muonParams,
                                                     adamwParams,
                                                     TrainingConfig.OptConfig.LearningRateBase,
                                                     TrainingConfig.OptConfig.WeightDecay,
                                                     TrainingConfig.OptConfig.Beta1,
                                                     true, 5,
                                                     TrainingConfig.OptConfig.Beta1,
                                                     TrainingConfig.OptConfig.Beta2,
                                                     1E-8f);
#if NOT
    public static MuonOptimizer MuonOptimizer(IEnumerable<Parameter> muonParams, IEnumerable<Parameter> adamwParams = null,  
                                              double lr = 1e-3, double wd = 0.1, double momentum = 0.95, 
                                              bool nesterov = true, int nsSteps = 5,
                                              double adamwBeta1 = 0.95, double adamwBeta2 = 0.95, double adamwEps = 1e-8)

        optimizer = new MuonOptimizer(TrainingConfig.OptConfig.LearningRateBase, 
                                      TrainingConfig.OptConfig.WeightDecay,
                                      muonParams,
                                      momentum: TrainingConfig.OptConfig.Beta1,
                                      nesterov: true, 
                                      nsSteps: 5,
                                      adamwParams,
                                      (TrainingConfig.OptConfig.Beta1, TrainingConfig.OptConfig.Beta2), 
                                      1E-8f);
#endif
        //optimizer = MuonOptimizer(adamWOptGroups);
      }
      else if (TrainingConfig.OptConfig.Optimizer == OptimizerType.NAdamW)
      {
        // TODO: Rerun NAdam after TorchSharp bug noted here is fixed: https://github.com/dotnet/TorchSharp/pull/1155
        // This looked very promising. But doesn't seem work with weight decay, need "NadamW" ?
        // Also try this PyTorch version of NadamW. Note that it flags one line in the code which
        // is the difference from the official PyTorch version. It may be best
        // to clone the PyTorch version and transplant this one line to get it working with weight decay.
        // https://github.com/runame/algorithmic-efficiency/blob/c0d8cd3819dcdada07a4b74bfba688e6f4fadbba/baselines/nadamw/pytorch/submission.py

        // N.B. Not yet implemented is per-parameter weight decay, etc.
        optimizer = NAdam(model.parameters(),
                          lr: TrainingConfig.OptConfig.LearningRateBase,
                          weight_decay: TrainingConfig.OptConfig.WeightDecay * 0, // <----------- NOTE DISABLED ----------------
                          beta1: TrainingConfig.OptConfig.Beta1,
                          beta2: TrainingConfig.OptConfig.Beta2);
      }
      else
      {
        throw new NotImplementedException(TrainingConfig.OptConfig.Optimizer.ToString());
      }


      if (startingCheckpointFN != null)
      {
        throw new NotImplementedException("Restore from checkpoint not yet implemented in C# backend (especially reloading optimizer)");
        Console.WriteLine("Loading model weights " + startingCheckpointFN);
        model.load(startingCheckpointFN);//
                                         //        optimizer.load(startingCheckpointFN.Replace("_model", "_opt"));
                                         //        currentPosNum = startingCheckpointLastNumPos;
      }


      LRScheduler scheduler = new LambdaLR(optimizer,
                                   delegate (int e)
                                   {
                                     float fractionComplete = (float)numPositionsReadFromTraining / TrainingConfig.OptConfig.NumTrainingPositions;
                                     float FRAC_START_DECAY = TrainingConfig.OptConfig.LRBeginDecayAtFractionComplete;

                                     float lrScale;
                                     if (numPositionsReadFromTraining < 20_000_000 && (fractionComplete < 0.02
                                                                                  || numPositionsReadFromTraining < 500_000))
                                     {
                                       lrScale = TrainingConfig.OptConfig.LRWarmupPhaseMultiplier;
                                     }
                                     else if (fractionComplete < FRAC_START_DECAY)
                                     {
                                       lrScale = 1.0f;
                                     }
                                     else
                                     {
                                       // # Once decay starts, LR multiplier is same as fraction remaining until end of training.
                                       lrScale = 1.0f - fractionComplete;
                                     }

                                     return MathF.Max(0.10f, MathF.Min(lrScale, 1));
                                   });

      Stopwatch sw = new Stopwatch();
      sw.Start();

      // Show number of model parameters.
      NumParameters = TorchSharpUtils.NumModuleParameters(model);
      if (!silentMode)
      {
        Console.WriteLine($"INFO: NUM_PARAMETERS {NumParameters}");
        Console.WriteLine();
      }

      CrossEntropyLoss valueLoss = CrossEntropyLoss();
      CrossEntropyLoss value2Loss = CrossEntropyLoss();
      CrossEntropyLoss policyLoss = CrossEntropyLoss();

      HuberLoss mlhLoss = new(0.5f, Reduction.Mean);
      HuberLoss uncLoss = new(0.5f, Reduction.Mean);
      MSELoss uncPolicyLoss = new();
      MSELoss qDeviationLowerLoss = new();
      MSELoss qDeviationUpperLoss = new();

      CeresTrainInitialization.PrepareToUseDevice(TrainingConfig.ExecConfig.Device);

      using (var d = NewDisposeScope())
      {
        scheduler.step();

        bool useNullOutput = anchorEvaluator != null; 
        consoleStatusTable = new(TrainingConfig.ExecConfig.ID, trainingSessionDescription, TrainingConfig.OptConfig.NumTrainingPositions, false, useNullOutput);

        consoleStatusTable.RunTraining(
          () => Train(TrainingConfig.ExecConfig.ID, model, anchorEvaluator, optimizer, 
                      policyLoss, valueLoss, value2Loss,
                      mlhLoss, uncLoss, uncPolicyLoss,
                      qDeviationLowerLoss, qDeviationUpperLoss,
                      null, null, // TO DO: Fill in value difference tensors
                      train,
                      tpgDataset, scheduler, tbWriter,
                      TrainingConfig.OptConfig.NumTrainingPositions,
                      ref numPositionsReadFromTraining, silentMode));
      }

      sw.Stop();

      if (!silentMode)
      {
        if (saveNetworkFileName != null)
        {
          Console.WriteLine($"INFO: TORCHSCRIPT_FILENAME {saveNetworkFileName}");
        }

        Console.WriteLine("INFO: EXIT_STATUS SUCCESS");
      }
    }


    /// <summary>
    /// Returns a array of AdamW.ParamGroups, 
    /// one group for which weight decay is applied and a group for which it is not.
    /// </summary>
    /// <param name="model"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    AdamW.ParamGroup[] GetOptimizerAdamWGroups(Module model)
    {
      HashSet<Type> whitelistWeightModules = [typeof(Linear), typeof(LayerSoftMoEBatchedDual), typeof(LayerMultiExpert)];
      HashSet<Type> blacklistWeightModules = [typeof(LayerNorm), typeof(Embedding), typeof(RMSNorm)];

      HashSet<string> decay = new();
      HashSet<string> noDecay = new();

      foreach ((string name, Module module) namedModule in model.named_modules())
      {
        string moduleName = namedModule.name;
        Module module = namedModule.module;

        foreach ((string name, Parameter parameter) namedParam in module.named_parameters())
        {
          // Skip non-updating parameters
          if (!namedParam.parameter.requires_grad)
          {
            continue;
          }

          string paramName = namedParam.name;
          Parameter param = namedParam.parameter;

          string fullParamName = string.IsNullOrEmpty(moduleName) ? paramName : $"{moduleName}.{paramName}";

          if (fullParamName.ToLower().Contains("embedding"))
          {
            noDecay.Add(fullParamName);
          }
          else if (fullParamName.ToLower().Contains("lora"))
          {
            noDecay.Add(fullParamName);
          }
          else if (paramName.ToLower().EndsWith("bias"))
          {
            noDecay.Add(fullParamName);
          }
          else if (blacklistWeightModules.Contains(module.GetType()))
          {
            noDecay.Add(fullParamName);
          }
          else if (whitelistWeightModules.Contains(module.GetType()))
          {
            decay.Add(fullParamName);
          }
        }
      }

      Dictionary<string, Parameter> paramDict = model.named_parameters().ToDictionary(p => p.name, p => p.parameter);
      IEnumerable<string> interParams = decay.Intersect(noDecay);
      IEnumerable<string> unionParams = decay.Union(noDecay);

      if (interParams.Any())
      {
        throw new InvalidOperationException($"Parameters {string.Join(", ", interParams)} appear in both decay/no_decay sets");
      }

      if (paramDict.Keys.Except(unionParams).Any())
      {
        throw new InvalidOperationException($"Parameters {string.Join(", ", paramDict.Keys.Except(unionParams))} were not fully partitioned into decay/no_decay sets");
      }

      AdamW.ParamGroup pgWD = new(decay.Select(pn => paramDict[pn]), AdamWOptionsWithWD(TrainingConfig.OptConfig.WeightDecay));
      AdamW.ParamGroup pgNoWD = new(noDecay.Select(pn => paramDict[pn]), AdamWOptionsWithWD(0));

      //      Console.WriteLine("Partitioned parameters into decay/no_decay sets:");
      //      Console.WriteLine($"  decay: {pgWD.Parameters.Count()}");
      //      Console.WriteLine($"  no_decay: {pgNoWD.Parameters.Count()}");
      return [pgNoWD, pgWD];
    }


    public void Dispose()
    {
      Model.Dispose();
      optimizer.Dispose();
      mlhTarget?.Dispose();
      uncTarget?.Dispose();
      valueTarget?.Dispose();
      policyTarget?.Dispose();
      maskedPolicyForLoss?.Dispose();
      value2Target?.Dispose();
      qDeviationLowerTarget?.Dispose();
      qDeviationUpperTarget?.Dispose();
      lossValueBatch?.Dispose();
      lossPolicyBatch?.Dispose();
      lossUNCPolicyBatch?.Dispose();
      lossMLHBatch?.Dispose();
      lossUNCBatch?.Dispose();
      lossValue2Batch?.Dispose();
      lossValueDBatch?.Dispose();
      lossValue2DBatch?.Dispose();
      lossTotal?.Dispose();
    }


    #region Training Monitoring

    /// <summary>
    /// Experimental code for monitoring progress while training via NNEvaluator.
    /// </summary>
    const bool TEST_EVAL_FROM_CSHARP = false;
    NNEvaluatorTorchsharp evaluatorCeres = null;
    public static PositionWithHistory TEST_PWH;


    public void DumpTestEvalStats()
    {
      List<float> qBaseline = new();
      List<float> qLoRA = new();
      List<float> qTPG = new();
      foreach (TPGRecord tpg in TrainingConfig.DataConfig.TPGFixedSet2)
      {
        Console.WriteLine();
        Console.WriteLine(tpg.SearchQ + " " + tpg.PolicyVector + " " + tpg.FinalPosition.FEN);
        NetTransformer transformer = (NetTransformer)Model;
        transformer.LoRAEnabled = false;
        NNEvaluatorResult resultBaseline = evaluatorCeres.Evaluate(tpg.ToPositionWithHistory());
        transformer.LoRAEnabled = true;
        NNEvaluatorResult lora = evaluatorCeres.Evaluate(tpg.ToPositionWithHistory());
        Console.WriteLine("Baseline : " + resultBaseline);
        Console.WriteLine("Ceres    : " + lora);

        qBaseline.Add(resultBaseline.V);
        qLoRA.Add(lora.V);
        qTPG.Add(tpg.SearchQ);
      }
      Console.WriteLine(StatUtils.RankCorrelation(qTPG.ToArray(), qBaseline.ToArray()) + " --> " +
                        StatUtils.RankCorrelation(qTPG.ToArray(), qLoRA.ToArray()));
      Console.WriteLine();
    }

    #endregion

    /// <summary>
    /// Computes cross-entropy between two distributions, each specified by logits.
    /// The shapes of logitsLeft and logitsRight should match, e.g. [batchSize, numClasses].
    /// Returns a scalar Tensor (the mean cross-entropy over the batch).
    /// </summary>
    public static Tensor CrossEntropyTwoLogits(Tensor inputs, Tensor targets)
    {
      // 1) Convert "left" logits to probabilities via softmax
      Tensor pLeft = softmax(inputs, dim: -1);

      // 2) Convert "right" logits to log probabilities
      Tensor logPRight = torch.nn.functional.log_softmax(targets, dim: -1);

      // 3) Compute cross-entropy: - sum_i [ p_left[i] * log_p_right[i] ]
      //    sum along the classes dimension (dim = -1), result is [batch], then average
      Tensor crossEntropyPerSample = pLeft.mul(logPRight).sum(dim: -1).neg();
      Tensor loss = crossEntropyPerSample.mean();

      return loss;
    }
  }



}
