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
    float lossMLHAdjT;
    float lossUNCAdjRunning;
    float lossUNCAdjT;

    float thisLossAdjRunning;

    private void DumpTrainingStatsToConsole(string configID, Tensor value, Tensor policy, Tensor mlh, Tensor unc, ref long numRead)
    {
      float[] predictionsMLHAll = null;
      float[] predictionsUNCAll = null;


      if ((object)mlh != null)
      {
        predictionsMLHAll = mlh.to(ScalarType.Float32).cpu().data<float>().ToArray();
      }

      if ((object)unc != null)
      {
        predictionsUNCAll = unc.to(ScalarType.Float32).cpu().data<float>().ToArray();
      }

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

      // update exponential averages
      const float WT_CUR = 0.15f; // smoothing to include decayed prior values
      const int NUM_SKIP = 10; // first few extremely large errors can create long tailed bias

      //            float lossPolicyAdj = NetConfig.INCLUDE_POLICY ? 100 * MathF.Sqrt(lossPolicyBatch.ToSingle() / NetConfig.BATCH_SIZE) : 0;
      float[] evalsMLH = null;
      if ((object)lossMLHBatch != null)
      {
        //evalsMLH = mlhTarget.to(ScalarType.Float32).cpu().data<float>().ToArray();
        float lossMLHAdj = (object)lossMLHBatch == null ? 0 : lossMLHBatch.ToSingle();
        lossMLHAdjT = numStatusLines < NUM_SKIP ? lossMLHAdj : lossMLHAdj * WT_CUR + lossMLHAdjT * (1.0f - WT_CUR);
      }

      float[] evalsUNC = null;
      if ((object)lossUNCBatch != null)
      {
        //evalsUNC = uncTarget.to(ScalarType.Float32).cpu().data<float>().ToArray();
        float lossUNCAdj = (object)lossUNCBatch == null ? 0 : lossUNCBatch.ToSingle();
        lossUNCAdjT = numStatusLines < NUM_SKIP ? lossUNCAdj : lossUNCAdj * WT_CUR + lossUNCAdjT * (1.0f - WT_CUR);
      }

      float SmoothedValue(float currentValue, float runningAverageValue)
      {
        return numStatusLines < NUM_SKIP ? currentValue
                                         : currentValue * WT_CUR + runningAverageValue * (1.0f - WT_CUR);
      }

      lossValueAdjRunning = SmoothedValue(lossValueAdj, lossValueAdjRunning);
      lossPolicyAdjRunning = SmoothedValue(lossPolicyAdj, lossPolicyAdjRunning);
      valueAccAdjRunning = SmoothedValue(valueAccuracy, valueAccAdjRunning);
      policyAccAdjRunning = SmoothedValue(policyAccuracy, policyAccAdjRunning);
      lossMLHAdjRunning = SmoothedValue(lossMLHAdjT, lossMLHAdjRunning);
      lossUNCAdjRunning = SmoothedValue(lossUNCAdjT, lossUNCAdjRunning);

      numStatusLines++;

      double elapsedSec = (DateTime.Now - timeStartTraining).TotalSeconds;
      double curLR = optimizer.ParamGroups.FirstOrDefault().LearningRate;

      thisLossAdjRunning = lossValueAdjRunning + lossPolicyAdjRunning
                          + lossMLHAdjRunning + lossUNCAdjRunning;

      // Write log statistics.
      tbWriter?.AddScalars((int)(numRead / 1024), // show in K to avoid overflow since must fit as int
                           ("lossVP", thisLossAdjRunning),
                           ("val", lossValueAdjRunning),
                           ("val_acc", valueAccAdjRunning),
                           ("pol", lossPolicyAdjRunning),
                           ("mlh", lossMLHAdjT),
                           ("unc", lossUNCAdjT),
                           ("acc", policyAccAdjRunning * 100));

      long positionsProcessed = batchId * OptimizationBatchSizeForward;
      string positionsAndBatchStr = (OptimizationBatchSizeForward != OptimizationBatchSizeBackward)
        ? $"[{positionsProcessed:N0} / {OptimizationBatchSizeForward}f / {OptimizationBatchSizeBackward}b] "
        : $"[{positionsProcessed:N0} / {OptimizationBatchSizeForward}] ";

      consoleStatusTable.UpdateInfo(DateTime.Now, configID, (float)elapsedSec, numRead, thisLossAdjRunning,
                                    lossValueAdjRunning, valueAccAdjRunning, lossPolicyAdjRunning, policyAccAdjRunning, (float)curLR);

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
