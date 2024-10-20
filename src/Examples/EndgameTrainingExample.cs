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
using System.IO;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.UserSettings;
using CeresTrain.PositionGenerators;
using CeresTrain.Networks.Transformer;
using CeresTrain.Trainer;
using CeresTrain.UserSettings;
using CeresTrain.NNEvaluators;
using Ceres.Chess.NNEvaluators.Ceres;

#endregion 

namespace CeresTrain.Examples
{
  /// <summary>
  /// Example code that demonstrates:
  ///   - training a neural network on endgame positions
  ///   - testing accuracy of trained net on a set of positions
  ///   - running a tournament of trained net against an LC0 network
  ///   - running a UCI loop with trained net (to allow interactive play)
  /// </summary>
  public partial class EndgameTrainingExample
  {
    /// Define set of supported endgame pieces and associated random position generator.
    static string PIECES_STRING = "KNNkpp,KPPknn";
    public static PositionGenerator randPosGenerator => configData.PositionGenerator;

    const int GPU_ID = 0;
    const string LCO_NET_ID = "811971"; // ID of LC0 network used for comparison

    // Define execution environment for training and testing (use CUDA device with our target GPU).
    static ConfigNetExecution configExec = new ConfigNetExecution("default", "CUDA", [GPU_ID]);

    // Define the network architecture.
    // Here we use a transformer with dimension 256, 8 layers, 8 attention heads, feedforward multiplier 1,
    // and some additional customizations specified in the with clause.
    static NetTransformerDef configTransformerDef = new NetTransformerDef(256, 8, 8, 1, NetTransformerDef.TransformerFeatures.Smolgen) with
    {
//      PreNorm = false,

//      SmolgenDimPerSquare = 8,
//      SmolgenDim = 64,

//      HeadWidthMultiplier=2,

//      SoftMoEConfig = new SoftMoEParams()  with 
//      { NumExperts = 16, MoEMode = SoftMoEParams.SoftMoEModeType.AddLinearSecondLayer,
//        NumSlotsPerExpert = 1, OnlyForAlternatingLayers = true,
//        UseBias = true, UseNormalization = false
//      }
    };
  

  // Define preferred options for monitoring training progress.
  static ConfigMonitoring configMonitoring = new ConfigMonitoring() with
    {
      DumpInternalActivationStatistics = false
    };

    const long NUM_TRAINING_POSITIONS = 10_000_000; // adjust to control training duration

    // Define training data configuration, specify size of batches generated by training data loader.
    static ConfigData configData = new ConfigData(new PositionGeneratorRandomFromPieces(PIECES_STRING));

    // Define the optimization parameters such as effective batch size, learning rate base and schedule.
    static ConfigOptimization configOptimization = new ConfigOptimization(NUM_TRAINING_POSITIONS) with
    {
      NumTrainingPositions = NUM_TRAINING_POSITIONS,
//      BatchSizeBackwardPass = 2048, // use gradient accumulation to make effective batch size larger
      LearningRateBase = 5E-4f, // baseline learning rate (scaled as on next line). Larger nets may need lower values.
      LossValueMultiplier = 1f, // curiously lower weight on value performs better despite scale of value loss much smaller than policy,
      GradientClipLevel= 2
    };

    // Finally, construct top-level training configuration object based on above.
    public static ConfigTraining trainingConfig => new ConfigTraining(configExec, configTransformerDef,
                                                                      configData, configOptimization, configMonitoring);



    /// <summary>
    /// Main entry point for example code.
    /// </summary>
    public static void RunAllTests()
    {
      CheckPrerequisites();

      // Train a network (on KRPkrp endgames).
      TrainingResultSummary result = TrainNet("Test1", PIECES_STRING, 10_000_000);

      NNEvaluatorInferenceEngineType engineType = NNEvaluatorInferenceEngineType.CSharpViaTorchscript;

      // Test accuracy of trained network on a set of random endgame positions.
      NNEvaluatorOptionsCeres options = default; // TODO: fill this in
      CeresNetEvaluation.TestAccuracyOnPositions(randPosGenerator, null, CeresNetEvaluation.GetNNEvaluator(engineType, configTransformerDef, 0, in configExec, result.TorchscriptFileName, true, options), null, result);

      // Run tournaments (value/policy) between the trained network and an LC0 reference network.
      CeresNetEvaluation.RunTournament(engineType, configTransformerDef, in configExec, result.TorchscriptFileName, LCO_NET_ID, "GPU:0", randPosGenerator, SearchLimit.BestValueMove, 50);
      CeresNetEvaluation.RunTournament(engineType, configTransformerDef, in configExec, result.TorchscriptFileName, LCO_NET_ID, "GPU:0", randPosGenerator, SearchLimit.NodesPerMove(1), 50);

      // Run an interactive UCI session in the console using the trained network.
      PositionGeneratorRandomFromPieces posGenerator = new(PIECES_STRING);
      CeresNetEvaluation.RunUCILoop(configTransformerDef, in configExec, result.TorchscriptFileName, LCO_NET_ID, "GPU:0", posGenerator);
    }


    /// <summary>
    /// Train a neural network from scratch, using tablebase positions as training data.
    /// </summary>
    /// <returns></returns>
    public static TrainingResultSummary TrainNet(string configName, string piecesStr, long numPos)
    {
      if (numPos == 0)
      {
        throw new Exception("numPos must be > 0");
      }

      TrainingHelpers.AdjustAndLoadConfig(configName, piecesStr, null, null);

      // Construct a trainer object using this configuration.
      string description = $"Training {piecesStr} with {numPos} positions";
      TrainingStatusTable statusTable = new TrainingStatusTable(trainingConfig.ExecConfig.ID, description, trainingConfig.OptConfig.NumTrainingPositions, false);
      CeresTrainCommandTrain boardTrain = new(trainingConfig, description, statusTable);

      // Run training session.
      TrainingResultSummary result = boardTrain.DoTrain("EndgameTrainingExample");

      string resultsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "results");
      string resultsFileName = Path.Combine(resultsDir, configName) + "_results.json";

      Console.WriteLine("");
      Console.WriteLine($"Training complete {result.NumTrainingPositions:N0} steps in {result.TrainingTime}");

      return result;
    }


    
    #region Helper methods

    /// <summary>
    /// Verifies that the system prerequisites for running the example code are met.
    /// </summary>
    static void CheckPrerequisites()
    {
      string tbPath = CeresUserSettingsManager.Settings.SyzygyPath;
      if (tbPath == null || !Path.Exists(tbPath))
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Ceres.json must contain SyzygyPath entry referencing a valid directory, got: " + tbPath);
      }
    }

    #endregion
  }
}
