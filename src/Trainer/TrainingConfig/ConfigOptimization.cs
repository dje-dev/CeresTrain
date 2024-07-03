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

using System.Text.Json.Serialization;

#endregion

namespace CeresTrain.Trainer
{
  public enum OptimizerType
  {
    /// <summary>
    /// Stochastic gradient descent optimizer.
    /// </summary>
    SGD,

    /// <summary>
    ///  Adam optimizer.
    /// </summary>
    AdamW,

    /// <summary>
    /// Adam optimizer with 8-bit quantization.
    /// </summary>
    AdamW8bit,

    /// <summary>
    /// Nadam optimizer (with decoupled weight decay).
    /// 
    /// The paper "Benchmarking Neural Network Training Algorithms" Dahl et al. 2023
    /// notes that NAdam matches our outperforms all other tested optimizers in all configurations
    /// (for transformer network).
    /// </summary>
    NAdamW,

    /// <summary>
    /// Lion optimizer.
    /// </summary>
    LION,
  }


  /// <summary>
  /// Parameters related to optimization.
  /// </summary>
  public readonly record struct ConfigOptimization
  {
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="numTrainingPositions"></param>
    public ConfigOptimization(long numTrainingPositions) : this()
    {
      NumTrainingPositions = numTrainingPositions;
    }

    /// <summary>
    /// Default constructor for deserialization.
    /// </summary>
    [JsonConstructorAttribute]
    public ConfigOptimization()
    {
    }

    /// <summary>
    /// Number of training positions to use before halting training.
    /// </summary>
    public readonly long NumTrainingPositions { get; init; } = int.MaxValue;

    /// <summary>
    /// Batch size used each forward pass.
    /// </summary>
    public readonly int BatchSizeForwardPass { get; init; } = 2048;

    /// <summary>
    /// Batch size used for each backward pass (possibly larger than BatchSizeForwardPass if accumulating gradients).
    /// </summary>
    public readonly int BatchSizeBackwardPass { get; init; } = 2048;

    /// <summary>
    /// Type of optimizer.
    /// </summary>
    public readonly OptimizerType Optimizer { get; init; } = OptimizerType.AdamW;

    /// <summary>
    /// Optional name of file containing the starting checkpoint from which training will be resumed.
    /// </summary>
    public readonly string StartingCheckpointFN { get; init; }


    /// <summary>
    /// If StartingCheckpointFN is not null, this is the number of positions which were trained as of the end of that checkpoint file.
    /// </summary>
    public readonly long StartingCheckpointLastPosNum { get; init; }
    
    /// <summary>
    /// String to be used for model argument of the PyTorch compile method (or null for no compile).
    /// Valid values: "default", "reduce-overhead", or "max-autotune".
    /// </summary>
    public readonly string PyTorchCompileMode { get; init; } = "max-autotune";

    /// <summary>
    /// Weight decay coefficient for the optimizer.
    /// </summary>
    public readonly float WeightDecay { get; init; } = 0.01f;

    /// <summary>
    /// Maximum learning rate to be used with the optimizer.
    /// </summary>
    public readonly float LearningRateBase { get; init; } = 2E-4f;

    /// <summary>
    /// Fraction complete (between 0 and 1) at which
    /// scaling down of the LearningRateBase begins 
    /// (linearly from LearningRateBase to a fixed minimum value of 0.10).
    /// </summary>
    public readonly float LRBeginDecayAtFractionComplete { get; init; } = 0.5f;

    /// <summary>
    /// Beta 1 coefficient used with optimizers such as Adam, AdamW, or NAdamW.
    /// </summary>
    public readonly float Beta1 { get; init; } = 0.90f;

    /// <summary>
    /// Beta 2 coefficient used with optimizers such as Adam, AdamW, or NAdamW.
    /// </summary>
    public readonly float Beta2 { get; init; } = 0.98f;

    /// <summary>
    /// Value at which gradients are clipped on each optimizer step (clipping is disabled if this value is 0.0).
    /// </summary>
    public readonly float GradientClipLevel { get; init; } = 2.0f;

    #region Loss multipliers

    /// <summary>
    /// Scaling multiplier to be applied to primary value loss term.
    /// </summary>
    public readonly float LossValueMultiplier { get; init; } = 1.0f;

    /// <summary>
    /// Scaling multiplier to be applied to secondary value loss term.
    /// Typically a lower coefficient is used here because it is very noisy.
    /// </summary>
    public readonly float LossValue2Multiplier { get; init; } = 0.3f;

    /// <summary>
    /// Scaling multiplier to be applied to policy loss term.
    /// </summary>
    public readonly float LossPolicyMultiplier { get; init; } = 1.0f;

    /// <summary>
    /// Scaling multiplier to be applied to MLH loss term.
    /// </summary>
    public readonly float LossMLHMultiplier { get; init; } = 0.0f;

    /// <summary>
    /// Scaling multiplier to be applied to value head uncertainty loss term.
    /// Coefficient typically small due to low importance in gameplay and relatively high noise.
    /// </summary>
    public readonly float LossUNCMultiplier { get; init; } = 0.01f;

    /// <summary>
    /// Scaling multiplier to be applied to estimates of lower and upper deviation bounds of forward Q.
    /// Coefficient typically small due to low importance in gameplay and relatively high noise.
    /// </summary>
    public readonly float LossQDeviationMultiplier { get; init; } = 0.01f;

    /// <summary>
    /// Scaling multiplier to be applied to policy uncertainty term.
    /// </summary>
    public readonly float LossUncertaintyPolicyMultiplier { get; init; } = 0.02f;


    /// <summary>
    /// Scaling multiplier to be applied to difference in value scores between consecutive positions.
    /// This acts as consistency regularizer.
    /// Seems to have small positlve benefit, especially for action head accuracy.
    /// </summary>
    public readonly float LossValueDMultiplier { get; init; } = 0.1f;

    /// <summary>
    /// Scaling multiplier to be applied to difference in value2 scores between consecutive positions.
    /// This acts as consistency regularizer.
    /// Benefit is unclear for Value2 because this training targtet is so noisy.
    /// </summary>
    public readonly float LossValue2DMultiplier { get; init; } = 0.0f;

    /// <summary>
    /// Loss weight applied to error in action prediction (relative to actual value2 from position).
    /// </summary>
    public readonly float LossActionMultiplier { get; init; } = 0.3f;

    /// <summary>
    /// Scaling multiplier to be applied to action value uncertainty term.
    /// </summary>
    public readonly float LossActionUncertaintyMultiplier { get; init; } = 0.01f;

    #endregion

    /// <summary>
    /// Reserved value used for debugging/experimentation to turn on a possible ad hoc test/diagnostic feature.
    /// </summary>
    public readonly float TestValue { get; init; } = 0;
  }
}
