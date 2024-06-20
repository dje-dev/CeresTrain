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
using CeresTrain.Trainer;
using CeresTrain.Networks.Transformer;
using CeresTrain.TrainCommands;
using CeresTrain.Networks.SoftMoE;

using static CeresTrain.Networks.Transformer.NetTransformerDef;
using System.Collections.Generic;

#endregion

namespace CeresTrain.Examples
{
  /// <summary>
  /// Sample code for running 2 concurrent training sessions on a remote Linux host (possibly via WSL on Windows).
  /// 
  /// The Linux host must be acessible via SSH with keys already configured to allow passwordless login.
  /// 
  /// A file CeresTrainHosts.json must exist in the root of the CeresTrain output directory,
  /// with an entry corresponding to the host used below. Example:
  /// [
  ///   {
  ///    "HostName": "server1",
  ///    "UserName": "myusername",
  ///    "CeresTrainPyDir": "/home/myusename/dev/CeresTrain/src/CeresTrainPy",
  ///    "PathToOutputFromHost": "/home/CeresTrain/outputs",
  ///   },
  /// ]
  /// </summary>
  public static class LaunchDistributedTraining
  {
    const string HOST_ID = "wsl";
    const string TRAINING_FILES_DIR = @"/mnt/e/test/test_tpg"; // location of the TPG files (from pespective of host)

    const string ROOT_RUN_ID = "192_6_4_4_500mm";
    const long NUM_TRAINING_POS = 100_000_000;

    // A very small network for test purposes.
    const int EMBEDDING_DIM = 192;
    const int NUM_LAYERS = 6;
    const int NUM_HEADS = 4;
    const int FFN_DIM = 4;

    /// <summary>
    /// Writes out the config files based on the C# definitions below.
    /// Then launches execution on one or more hosts and writes Console output to 
    /// continuously monitor the progress of each training session.
    /// </summary>
    public static void Run()
    {
      List<TrainingSessionSpecification> trainingRuns =
      [
        // RUN1: Baseline run (Mish activation for FFN) on GPU 0.
        MakeSessionSpec(ROOT_RUN_ID + "_mish", HOST_ID, [0], TRAINING_FILES_DIR, NUM_TRAINING_POS),

        // RUN2: Variant with squared ReLU activation for FFN) on GPU 1.
        MakeSessionSpec(ROOT_RUN_ID + "_sqrelu", HOST_ID, [1], TRAINING_FILES_DIR, NUM_TRAINING_POS,
                        netDefModifier: d => d with {FFNActivationType = ActivationType.ReLUSquared }),
      ];

      // Run all the sessions in concurrently (or choose option to only write configs).
      TrainingResultSummary[] summaries = CeresTrainBatchExecutor.RunSessions
        (
          CeresTrainBatchExecutor.BatchExecutorMode.WriteConfigsAndRunTrainingSessions,
          trainingRuns
        );

      System.Environment.Exit(3);
    }


    public static TrainingSessionSpecification MakeSessionSpec
      (string id,
       string hostID,
       int[] hostGPUs,
       string trainingFilesDir,
       long numTrainingPositions,
       Func<NetTransformerDef, NetTransformerDef> netDefModifier = null,
       Func<ConfigOptimization, ConfigOptimization> optConfigModifier = null,
       Func<ConfigNetExecution, ConfigNetExecution> execConfigModifier = null,
       Func<ConfigData, ConfigData> dataConfigModifier = null)
    {
      // Replace null modifiers with identity functions.  
      if (netDefModifier == null) { netDefModifier = d => d; }
      if (optConfigModifier == null) { optConfigModifier = d => d; }
      if (execConfigModifier == null) { execConfigModifier = d => d; }
      if (dataConfigModifier == null) { dataConfigModifier = d => d; }


      const bool MULTIBOARD = false;
      const int numExperts = 0;
      SoftMoEParams moeConfig = new SoftMoEParams() with
      {
        MoEMode = numExperts == 0 ? SoftMoEParams.SoftMoEModeType.None : SoftMoEParams.SoftMoEModeType.AddLinearSecondLayer,
        NumExperts = numExperts,
        OnlyForAlternatingLayers = true,
        NumSlotsPerExpert = 1,
        UseBias = true
      };



      ConfigTraining config = new ConfigTraining() with
      {
        ExecConfig = execConfigModifier(new ConfigNetExecution() with
        {
          ID = "C",
          UseFP8 = false,
          RunInDocker = false, // CeresTrainHosts.json needs to be properly configured
          TestFlag = false,
        }),

        DataConfig = dataConfigModifier(new ConfigData() with
        {
          FractionQ = 1f, // Value1 will be the pure search Q result
          SourceType = ConfigData.DataSourceType.PreprocessedFromTAR,
          TrainingFilesDirectory = trainingFilesDir,
        }),

        NetDefConfig = netDefModifier(new NetTransformerDef(EMBEDDING_DIM, NUM_LAYERS, NUM_HEADS, FFN_DIM, TransformerFeatures.None) with
        {
          DenseFormer = false,
          DeepNorm = false,

          AttentionMultiplier = 1,
          DualAttentionMode = DualAttentionModeType.None,
          UseRelBias = false,

          SmolgenDim = 0,
          UseRPE = true,

          PriorStateDim = MULTIBOARD ? 4 : 0,
          SoftMoEConfig = moeConfig,

          NormType = NormalizationType.RMSNorm,
          FFNActivationType = ActivationType.Mish,

          HeadsActivationType = ActivationType.Mish,
          HeadWidthMultiplier = 4, // remarkably, using 8 learns faster only at first then much worse, and using 2 is just plain worse
        }),

        OptConfig = optConfigModifier(new ConfigOptimization() with
        {
          NumTrainingPositions = numTrainingPositions,

          Optimizer = OptimizerType.AdamW,
          LearningRateBase = 3E-4f,
          LRBeginDecayAtFractionComplete = 0.5f,

          WeightDecay = 0.01f, // lower clearly better, but at extreme low levels can cause divergence or FP16 inference failure
          GradientClipLevel = 2,

          BatchSizeBackwardPass = 4096,
          BatchSizeForwardPass = 4096,

          LossPolicyMultiplier = 1,
          LossValueMultiplier = 1, // using 1 compared to 0.25 better by about +10 Elo @500mm with D512
          LossValue2Multiplier = 0.3f, // noisy target, so use a small value

          LossUNCMultiplier = 0.02f, // noisy target, and a less important target to predict
          LossQDeviationMultiplier = 0.01f, // noisy target, less important to predict

          LossMLHMultiplier = 0 * 0.05f, // not highly useful and slightly degreade results

          LossValue2DMultiplier = 0, // was deleterious, V2 is too noisy

          LossUncertaintyPolicyMultiplier = 0.02f, // somewhat noisy, but potentially important  in gameplay for temperature scaling

          LossActionMultiplier = MULTIBOARD ? 0.3f : 0.0f, // effective weight is lower because action losses applied only on subset of boards
          LossValueDMultiplier = MULTIBOARD ? 0.1f : 0.0f, // seems to speed convergence, improve action head
          LossActionUncertaintyMultiplier = MULTIBOARD ? 0.02f : 0.00f,

          PyTorchCompileMode = "max-autotune" // Sometimes "reduce-overhead" is as fast or faster (on A100 especially)
        }),

        MonitoringConfig = new ConfigMonitoring() with { },
      };


      return new TrainingSessionSpecification(id, CeresTrainHostConfig.RegisteredConfigs[hostID], hostGPUs, config);
    }

  }

}
