#region Using directives

using System;
using System.Linq;

using TorchSharp;
using static TorchSharp.torch;

using Ceres.Base.Math;

#endregion


namespace CeresTrain.Trainer
{
  public partial class CeresTrainCommandTrain : CeresTrainCommandBase
  {
    long lastReadReported;
    float priorLossAdjRunning;
    float bestLossAdjVP = float.MaxValue;

    int numStatusLines;
    float lossValueAdjRunning;
    float valueAccAdjRunning;
    float lossPolicyAdjRunning;
    float policyAccAdjRunning;
    float lossMLHAdjRunning;
    float lossUNCAdjRunning;

    float lossValue2AdjRunning;
    float lossQDeviationLowerAdjRunning;
    float lossQDeviationUpperAdjRunning;

    float thisLossAdjRunning;

    private void DumpTrainingStatsToConsole(string configID, Tensor value, Tensor policy, ref long numRead)
    {
      // Compare indices of max values to check if max value is at same index as in target.
      Tensor valueMatches = torch.max(value, 1).indexes.eq(torch.max(valueTarget, 1).indexes);
      float valueAccuracy = (valueMatches.sum().to_type(ScalarType.Float32)).item<float>() / value.size(0);

      // each (max_scores, max_idx_class)
      (Tensor values, Tensor indexes) maxTarget = policyTarget.max(dim: 1);
      (Tensor values, Tensor indexes) maxPredicted = maskedPolicyForLoss.max(dim: 1);
      //var errorArray = new TensorAccessor<float>(lossPolicyBatch.to(ScalarType.Float32).cpu()).ToArray();
      float policyAccuracy = (maxTarget.indexes == maxPredicted.indexes).sum().cpu().ReadCpuValue<long>(0) / (float)OptimizationBatchSizeForward;

      lastReadReported = numRead;
      timeLastSave = PossiblySaveNetwork(Model, optimizer, timeLastSave);
      timeLastDump = DateTime.Now;

      float lossValueAdj = TrainingConfig.OptConfig.LossValueMultiplier == 0 ? 0 : lossValueBatch.ToSingle();
      float lossPolicyAdj = TrainingConfig.OptConfig.LossPolicyMultiplier == 0 ? 0 : lossPolicyBatch.ToSingle();
      float lossMLHAdj = TrainingConfig.OptConfig.LossMLHMultiplier == 0 ? 0 : lossMLHBatch.ToSingle();
      float lossUNCAdj = TrainingConfig.OptConfig.LossUNCMultiplier == 0 ? 0 : lossUNCBatch.ToSingle();

      float lossValue2Adj = TrainingConfig.OptConfig.LossValue2Multiplier == 0 ? 0 : lossValue2Batch.ToSingle();
      float lossQDeviationLowerAdj = TrainingConfig.OptConfig.LossQDeviationMultiplier == 0 ? 0 : lossQDeviationLowerBatch.ToSingle();
      float lossQDeviationUpperAdj = TrainingConfig.OptConfig.LossQDeviationMultiplier == 0 ? 0 : lossQDeviationUpperBatch.ToSingle();


      // update exponential averages
      const float WT_CUR = 0.15f; // smoothing to include decayed prior values
      const int NUM_SKIP = 10; // first few extremely large errors can create long tailed bias

      float SmoothedValue(float currentValue, float runningAverageValue)
      {
        return numStatusLines < NUM_SKIP ? currentValue
                                         : currentValue * WT_CUR + runningAverageValue * (1.0f - WT_CUR);
      }

      lossValueAdjRunning = SmoothedValue(lossValueAdj, lossValueAdjRunning);
      lossPolicyAdjRunning = SmoothedValue(lossPolicyAdj, lossPolicyAdjRunning);
      valueAccAdjRunning = SmoothedValue(valueAccuracy, valueAccAdjRunning);
      policyAccAdjRunning = SmoothedValue(policyAccuracy, policyAccAdjRunning);
      lossMLHAdjRunning = SmoothedValue(lossMLHAdj, lossMLHAdjRunning);
      lossUNCAdjRunning = SmoothedValue(lossUNCAdj, lossUNCAdjRunning);

      lossValue2AdjRunning = SmoothedValue(lossValue2Adj, lossValue2AdjRunning);
      lossQDeviationLowerAdjRunning = SmoothedValue(lossQDeviationLowerAdj, lossQDeviationLowerAdjRunning);
      lossQDeviationUpperAdjRunning = SmoothedValue(lossQDeviationUpperAdj, lossQDeviationUpperAdjRunning);


      numStatusLines++;

      double elapsedSec = (DateTime.Now - timeStartTraining).TotalSeconds;
      double curLR = optimizer.ParamGroups.FirstOrDefault().LearningRate;

      thisLossAdjRunning = lossValueAdjRunning  * TrainingConfig.OptConfig.LossValueMultiplier
                          + lossPolicyAdjRunning * TrainingConfig.OptConfig.LossPolicyMultiplier
                          + lossMLHAdjRunning * TrainingConfig.OptConfig.LossMLHMultiplier
                          + lossUNCAdjRunning * TrainingConfig.OptConfig.LossUNCMultiplier
                          + lossValue2AdjRunning * TrainingConfig.OptConfig.LossValue2Multiplier
                          + lossQDeviationLowerAdjRunning  * TrainingConfig.OptConfig.LossQDeviationMultiplier
                          + lossQDeviationUpperAdjRunning *  TrainingConfig.OptConfig.LossQDeviationMultiplier;

      // Write log statistics.
      tbWriter?.AddScalars((int)(numRead / 1024), // show in K to avoid overflow since must fit as int
                           ("lossVP", thisLossAdjRunning),
                           ("val", lossValueAdjRunning),
                           ("val_acc", valueAccAdjRunning),
                           ("pol", lossPolicyAdjRunning),
                           ("mlh", lossMLHAdjRunning),
                           ("unc", lossUNCAdjRunning),
                           ("val2", lossValue2AdjRunning),
                           ("qDevL", lossQDeviationLowerAdjRunning),
                           ("qDevU", lossQDeviationUpperAdjRunning),
                           ("acc", policyAccAdjRunning * 100));

      long positionsProcessed = batchId * OptimizationBatchSizeForward;
      string positionsAndBatchStr = (OptimizationBatchSizeForward != OptimizationBatchSizeBackward)
        ? $"[{positionsProcessed:N0} / {OptimizationBatchSizeForward}f / {OptimizationBatchSizeBackward}b] "
        : $"[{positionsProcessed:N0} / {OptimizationBatchSizeForward}] ";

      consoleStatusTable.UpdateInfo(DateTime.Now, configID, (float)elapsedSec, numRead, thisLossAdjRunning,
                                    lossValueAdjRunning, valueAccAdjRunning, lossPolicyAdjRunning, policyAccAdjRunning, 
                                    lossMLHAdjRunning, lossUNCAdjRunning,
                                    lossValue2AdjRunning, lossQDeviationLowerAdjRunning, lossQDeviationUpperAdjRunning,
                                    (float)curLR);

      // Save network if best seen so far (on total loss) unless very early in training.
      const float BEST_MIN_LOSS_IMPROVEMENT = 0.01f;
      if (positionsProcessed > 10_000_000
       && bestLossAdjVP - thisLossAdjRunning > BEST_MIN_LOSS_IMPROVEMENT)
      {
        PossiblySaveNetwork(Model, optimizer, timeLastSave, true);
        bestLossAdjVP = thisLossAdjRunning;
      }

      // Do actual monitoring inside a try/catch block so it won't cause training to fail.
      try
      {
        monitor.DoMonitoring(Model, numRead);
      }
      catch (Exception ex)
      {
        Console.WriteLine("Exception in CeresTrainerMonitor: " + ex);
      }

      priorLossAdjRunning = thisLossAdjRunning;
    }

  }

}
