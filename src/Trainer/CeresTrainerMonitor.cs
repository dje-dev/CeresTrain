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

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;
using Ceres.Features.Suites;
using Ceres.Features.Tournaments;

using CeresTrain.Networks;
using CeresTrain.NNEvaluators;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// 
  /// </summary>
  public class CeresTrainerMonitor
  {
    /// <summary>
    /// Parent Trainer which is being monitored.
    /// </summary>
    public readonly CeresTrainCommandTrain TrainerC;

    public CeresTrainerMonitor(CeresTrainCommandTrain trainer)
    {
      TrainerC = trainer;
    }


    /// <summary>
    /// Runs suite of test positions using current evaluator.
    /// </summary>
    /// <param name="trainingEvaluatorDef"></param>
    /// <param name="tsEval"></param>
    public void RunSuiteTest(NNEvaluatorDef trainingEvaluatorDef, NNEvaluatorTorchsharp tsEval)
    {
      ref readonly ConfigTraining trainer = ref TrainerC.TrainingConfig;

      NNEvaluatorFactory.Custom2Factory = (netID, gpuID, referenceEvaluator) => tsEval;
      GameEngineDef ged1 = new GameEngineDefCeres("TrainNet", trainingEvaluatorDef, null); // TODO: eliminate need to specify a valid network ID here
      EnginePlayerDef ceresEngineDef1 = new EnginePlayerDef(ged1, trainer.MonitoringConfig.TestSuiteSearchLimit);
      SuiteTestDef def = new SuiteTestDef("Test1", trainer.MonitoringConfig.TestSuiteFileName, ceresEngineDef1);
      def.MaxNumPositions = trainer.MonitoringConfig.TestSuiteMaxPositions;
      def.DumpEPDInfo = true;
      def.Output = new StringWriter();
      Console.Write("  ");
      def.Callback = (epd, correctnessScore, searchResult) =>
      {
        Console.Write(correctnessScore > 0 ? "+" : ".");
      };
      Console.WriteLine();
      SuiteTestRunner ser = new SuiteTestRunner(def);
      SuiteTestResult suiteResult = ser.Run(1, outputDetail: true, false, false);
      File.WriteAllText("train_last_suite_result.txt", def.Output.ToString());

      NNEvaluatorResult evalResult = tsEval.Evaluate(Position.StartPosition);
      Console.WriteLine("  suite: " + suiteResult.AvgScore1 + " " + " V: " + evalResult.V + " " + evalResult.Policy);

    }

    /// <summary>
    /// Runs tournament using ucrrent evaluator.
    /// </summary>
    /// <param name="trainingEvaluatorDef"></param>
    /// <param name="tsEval"></param>
    public void RunTournamentTests(NNEvaluatorDef trainingEvaluatorDef, NNEvaluatorTorchsharp tsEval)
    {
      ref readonly ConfigTraining trainer = ref TrainerC.TrainingConfig;
      NNEvaluatorDef compareEvaluatorDef = new NNEvaluatorDef(trainer.MonitoringConfig.CompareNetDef, trainer.MonitoringConfig.CompareDeviceDef);
      NNEvaluator compareEvaluator = NNEvaluatorFactory.BuildEvaluator(compareEvaluatorDef);

      GameEngineDefCeres engineTrain = new("Train", trainingEvaluatorDef);
      GameEngineDefCeres engineCompare = new(trainer.MonitoringConfig.CompareNetDef.NetworkID, compareEvaluatorDef);

      foreach (SearchLimit limit in new SearchLimit[] { SearchLimit.NodesPerMove(1),
                                                                  SearchLimit.BestValueMove })
      {
        TournamentDef tournDef = new TournamentDef("Training", [new EnginePlayerDef(engineTrain, limit),
          new EnginePlayerDef(engineCompare, limit)]);
        tournDef.OpeningsFileName = trainer.MonitoringConfig.CompareNetOpeningsFileName;
        tournDef.ShowGameMoves = false;
        tournDef.NumGamePairs = limit.Type == SearchLimitType.BestValueMove ? trainer.MonitoringConfig.CompareNetTestValueNumGamePairs
                                                                            : trainer.MonitoringConfig.CompareNetTestPolicyNumGamePairs;
        tournDef.Logger = new StringWriter();

        // Start disable Console output
        TextWriter originalConsoleOut = Console.Out;
        Console.SetOut(TextWriter.Null);
        TournamentManager runner = new TournamentManager(tournDef, 1);
        TournamentResultStats result = runner.RunTournament(null, enableCancelVialCtrlC: false);

        // Restore Console output
        Console.SetOut(originalConsoleOut);

        string descStr = limit.Type == SearchLimitType.BestValueMove ? "Value" : "Policy";
        File.WriteAllText($"train_last_{descStr.ToLower()}_result.txt", tournDef.Logger.ToString());
        Console.Write($"  {descStr}: ");
        Console.Write(result.ShortEloSummaryStr);
        Console.WriteLine();
      }
    }


    /// <summary>
    /// Runs a complete set of monitoring operations.
    /// </summary>
    /// <param name="model"></param>
    /// <param name="numPositions"></param>
    public void DoMonitoring(CeresNeuralNet model, long numPositions)
    {
      // Switch out of training mode, no need to track gradients for testing.
      model.train(false);

      // Do actual monitoring inside a try/catch block so it won't cause training to fail.
      try
      {
        DoExecMonitoring(model, numPositions);
      }
      catch (Exception ex)
      {
        Console.WriteLine("Exception in CeresTrainerMonitor: " + ex);
      }

      // Return to training mode.
      model.train(true);
    }


    DateTime lastCustomMonitorTime = DateTime.Now;


    /// <summary>
    /// 
    /// </summary>
    /// <param name="Trainer"></param>
    /// <param name="model"></param>
    void DoExecMonitoring(CeresNeuralNet model, long numPositions)
    {
      ref readonly ConfigTraining trainer = ref TrainerC.TrainingConfig;
      bool needEvaluator = trainer.MonitoringConfig.TestSuiteFileName != null
                        //                        || trainer.Monitoring.CustomMonitoringCallback != null
                        || trainer.MonitoringConfig.CompareNetTestPolicyNumGamePairs > 0
                        || trainer.MonitoringConfig.CompareNetTestValueNumGamePairs > 0;

      const string NET_DUMMY_ID = "703810";
      NNEvaluatorDef trainingEvaluatorDef = new NNEvaluatorDef(NNEvaluatorType.Custom2, NET_DUMMY_ID);
      NNEvaluatorTorchsharp tsEval = default;
      if (needEvaluator)
      {
        ConsoleUtils.InvokeNoConsoleOutput(() =>
        {
          throw new Exception("needs minor remediation to work with CeresNeuralNetwork not CeresTransformer in next line");
          //ModuleNNEvaluatorFromTorchScript evaluator = new ModuleNNEvaluatorFromTorchScript(model, trainer.ExecutionConfig);
          //tsEval = new NNEvaluatorTorchsharp(evaluator, new Device("CUDA:0"), bfloat16, true, false);// model.Config.DataType);
          //tsEval.Evaluate(Position.StartPosition); // warmup
        });
      }

      // Run custom monitoring step, if any
      if (trainer.MonitoringConfig.CustomMonitoringCallback != null)
      {
        double secondsSince = (DateTime.Now - lastCustomMonitorTime).TotalSeconds;
        if (secondsSince > trainer.MonitoringConfig.CustomMonitoringIntervalSeconds)
        {
          string prefixString = $"{DateTime.Now} {trainer.ExecConfig.ID} {numPositions:N0} ";
          (string consoleOutputString, string fileOutputString) = trainer.MonitoringConfig.CustomMonitoringCallback(model, trainer, trainingEvaluatorDef, tsEval);
          if (consoleOutputString != null)
          {
            Console.WriteLine(prefixString + consoleOutputString);
          }
          if (fileOutputString != null)
          {
            File.AppendAllText($"train_monitor_{trainer.ExecConfig.ID}.txt",
                               Environment.NewLine + " " + Environment.NewLine +
                               prefixString + Environment.NewLine + fileOutputString);
          }

          lastCustomMonitorTime = DateTime.Now;
        }
      }

      if (trainer.MonitoringConfig.TestSuiteFileName != null)
      {
        RunSuiteTest(trainingEvaluatorDef, tsEval);
      }

      if (trainer.MonitoringConfig.CompareNetTestPolicyNumGamePairs > 0
       || trainer.MonitoringConfig.CompareNetTestValueNumGamePairs > 0)
      {
        RunTournamentTests(trainingEvaluatorDef, tsEval);
      }

    }
  }

}
