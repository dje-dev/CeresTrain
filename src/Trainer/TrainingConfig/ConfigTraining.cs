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
using System.Linq;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Defines all parameters (data, optimization, monitoring, etc.) used in training.
  /// </summary>
  public readonly record struct ConfigTraining
  { 
    /// <summary>
    /// Default constructor.
    /// </summary>
    public ConfigTraining()
    {

    }

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="tag"></param>
    /// <param name="executionConfig"></param>
    /// <param name="networkDef"></param>
    /// <param name="trainingDataConfig"></param>
    /// <param name="optimizationConfig"></param>
    /// <param name="monitoringConfig"></param>
    public ConfigTraining(ConfigNetExecution executionConfig,
                          ICeresNeuralNetDef networkDef,
                          ConfigData trainingDataConfig,
                          ConfigOptimization optimizationConfig,
                          ConfigMonitoring monitoringConfig = default)
    {
      DataConfig = trainingDataConfig;
      ExecConfig = executionConfig;
      NetDefConfig = networkDef;
      OptConfig = optimizationConfig;
      MonitoringConfig = monitoringConfig;
    }

    /// <summary>
    /// Specification of source of training data and preprocessing options.
    /// </summary>
    public readonly ConfigData DataConfig { get; init; } = new ();

    /// <summary>
    /// Specification of general network configuration (data type, device, etc.).
    /// </summary>
    public readonly ConfigNetExecution ExecConfig { get; init; } = new();

    /// <summary>
    /// Specification of architecture of the model.
    /// </summary>
    public readonly ICeresNeuralNetDef NetDefConfig { get; init; }

    /// <summary>
    /// Parameters related to optimization.
    /// </summary>
    public readonly ConfigOptimization OptConfig { get; init; } = new();

    /// <summary>
    /// Preferences related to periodic monitoring of training loop.
    /// </summary>
    public readonly ConfigMonitoring MonitoringConfig { get; init; } = new();



    private static readonly string[] validPyTorchCompileModes = ["none", "max-autotune", "reduce-overhead", 
                                                                  "max-autotune-no-cudagraphs", "default"];

    /// <summary>
    /// Check if the configuration is valid.
    /// </summary>
    public void Validate(bool runningInProcess)
    {
      NetDefConfig.Validate();

      if (OptConfig.BatchSizeForwardPass > OptConfig.BatchSizeBackwardPass)
      {
        throw new Exception("OptimizationConfig.BatchSizeForwardPass cannot be greater than OptimizationConfig.BatchSizeBackwardPass.");
      }

      if (OptConfig.PyTorchCompileMode != null && 
        !(validPyTorchCompileModes).Contains(OptConfig.PyTorchCompileMode))
      {
        throw new Exception($"Invalid OptimizationConfig.PyTorchCompileMode: {OptConfig.PyTorchCompileMode}");
      } 

      if (!runningInProcess 
       && OptConfig.PyTorchCompileMode.Contains("max-autotune")
       && OptConfig.BatchSizeBackwardPass != OptConfig.BatchSizeForwardPass)
      {
        // As of PyTorch 2.1 yields low-level Exception, possible bug in PyTorch.
        //throw new Exception("OptimizationConfig.PyTorchCompileMode == \"max-autotune\" not functioning properly with gradient accumulation (forward batch size < backward batch size).");
      }
    }

    /// <summary>
    /// Returns string describing a training session using this configuration.
    /// </summary>
    /// <param name="host"></param>
    /// <param name="trainingDir"></param>
    /// <returns></returns>
    public string TrainingDescriptionStr(string host, string trainingDir) =>
        $"Train {ExecConfig.ID} on {(host == null ? Environment.MachineName : host)} "
        + $"for {OptConfig.NumTrainingPositions:N0} pos on ({string.Join(",", ExecConfig.DeviceIDs)}) "
        + $"data in {DataConfig.TrainingFilesDirectory} from {trainingDir}";

  }
}
