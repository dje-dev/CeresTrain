﻿#region License notice

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
  /// The Linux host must be accessible via SSH with keys already configured to allow passwordless login.
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
    // The following two constants must definitely be changed to match your environment.
    const string HOSTNAME       = "SERVER1"; // name of host on which execution will take place
    const string TPG_SOURCE_DIR =@"/mnt/f/TPG_DATA"; // source directory containing TPGs

    const string RUN_ID_BASE = "256_10_8_4"; // identification string for training run
    const long NUM_TRAINING_POS = 200_000_000; // number of training positions

    const int EMBEDDING_DIM = 256;
    const int NUM_LAYERS = 10;
    const int NUM_HEADS = 8;
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
        // Baseline run on GPU 0.
        MakeSessionSpec(RUN_ID_BASE + "_BASE", HOSTNAME, [0], TPG_SOURCE_DIR, NUM_TRAINING_POS,
          netDef => netDef with {  },
          optDef => optDef with { },
          execDef => execDef with { },
          dataDef => dataDef with { }
          ),

        // Modified run on GPU 1 (turn off the Smolgen feature).
        MakeSessionSpec(RUN_ID_BASE + "_NO_SMOLGEN", HOSTNAME, [1], TPG_SOURCE_DIR, NUM_TRAINING_POS,
          netDef => netDef with { SmolgenDim = 0 }, 
          optDef => optDef with {  },
          execDef => execDef with { },
          dataDef => dataDef with { }
          ),

      ];


      // Run all the sessions in concurrently (or choose option to only write configs).
      TrainingResultSummary[] summaries = CeresTrainBatchExecutor.RunSessions
        (
          CeresTrainBatchExecutor.BatchExecutorMode.WriteConfigsAndRunTrainingSessions,
          trainingRuns.ToArray()
        );

      System.Environment.Exit(3);
    }


    public static TrainingSessionSpecification MakeSessionSpec
      (string id,
       string hostID,
       int[] hostGPUs,
       string trainingFilesDir,
       long numTrainingPositions,
       Func<NetTransformerDef, NetTransformerDef> netDefModifier,
       Func<ConfigOptimization, ConfigOptimization> optConfigModifier,
       Func<ConfigNetExecution, ConfigNetExecution> execConfigModifier,
       Func<ConfigData, ConfigData> dataConfigModifier)
    {
      const bool MULTIBOARD = false; // experimental only

      ConfigTraining config = new ConfigTraining() with
      {
        ExecConfig = execConfigModifier(new ConfigNetExecution() with
        {
          ID = "SP",
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

          PriorStateDim = MULTIBOARD ? 4 : 0,
          SoftMoEConfig = default,

          NormType = NormalizationType.RMSNorm,
          FFNActivationType = ActivationType.Mish,

          NonLinearAttention = true,
          UseRPE = false,

          SmolgenDimPerSquare = 32,
          SmolgenDim = 256,
          SmolgenActivationType = ActivationType.Swish,

          HeadsActivationType = ActivationType.Mish,
          HeadWidthMultiplier = 4, // remarkably, using 8 learns faster only at first then much worse, and using 2 is just plain worse
        }),

        OptConfig = optConfigModifier(new ConfigOptimization() with
        {
          NumTrainingPositions = numTrainingPositions,
          CheckpointFrequencyNumPositions = 100_000_000,

          Optimizer = OptimizerType.SOAP,
          Beta1 = 0.95f,
          Beta2 = 0.99f, // large B2 helpful (0.99, 0.999, 0.9995) for long-duration training "How Does Critical Batch Size Scale..."

          LearningRateBase = 2E-4f,
          LRBeginDecayAtFractionComplete = 0.50f,

          WeightDecay = 0.005f, // lower clearly better, but at extreme low levels can cause divergence or FP16 inference failure
          GradientClipLevel = 1f, // The SOAP paper seems to suggest low (possibly even very low) clips levels outperform.

          BatchSizeBackwardPass = 1024 * 4,
          BatchSizeForwardPass = 1024 * 2,

          LossPolicyMultiplier = 1.5f,
          LossValueMultiplier = 1, // using 1 compared to 0.25 better by about +10 Elo @500mm with D512
          LossValue2Multiplier = 0.04f, // noisy target, so use a small value

          LossUNCMultiplier = 0.01f, // noisy target, and a less important target to predict
          LossQDeviationMultiplier = 0.02f, // somewhat noisy target, less important to predict

          LossMLHMultiplier = 0 * 0.05f, // not highly useful and slightly degrade results

          LossValue2DMultiplier = 0, // was deleterious, V2 is too noisy

          LossUncertaintyPolicyMultiplier = 0.01f, // quite noisy, but potentially important in gameplay for temperature scaling

          LossActionMultiplier = MULTIBOARD ? 0.3f : 0.0f, // effective weight is lower because action losses applied only on subset of boards
          LossValueDMultiplier = MULTIBOARD ? 0.1f : 0.0f, // seems to speed convergence, improve action head
          LossActionUncertaintyMultiplier = 0, // (may have a bug in the Python code) MULTIBOARD ? 0.02f : 0.00f,

          PyTorchCompileMode = "default" // for faster training (but slower startup), consider: "max-autotune-no-cudagraphs" 
        }),

        MonitoringConfig = new ConfigMonitoring() with { },
      };


      return new TrainingSessionSpecification(id, CeresTrainHostConfig.RegisteredConfigs[hostID], hostGPUs, config);
    }

  }

}
