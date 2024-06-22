#region Using directives

using Ceres.Chess.NNEvaluators;
using System;

#endregion

namespace CeresTrain.NNEvaluators
{

  /// <summary>
  /// Set of options for NNEvaluatorTorchsharp.
  /// </summary>
  public record NNEvaluatorTorchsharpOptions : NNEvaluatorOptions
  {
    /// <summary>
    /// If the prior state information should be used.
    /// </summary>
    public bool UsePriorState { get; init; } = false;

    /// <summary>
    /// If the action head should be used.
    /// </summary>
    public bool UseAction { get; init; } = false;

    /// <summary>
    /// Assumed magnitude (Q units) of adverse blunders that will follow in the game.
    /// </summary>
    public float QNegativeBlunders { get; init; } = 0;

    /// <summary>
    /// Assumed magnitude (Q units) of favorable blunders that will follow in the game.
    /// </summary>
    public float QPositiveBlunders { get; init; } = 0;

    /// <summary>
    /// The order of the power mean used to combine the value heads.
    /// Default value of 1 corresponds to the arithmetic mean (0 for geometric).
    /// </summary>
    public float ValueHeadAveragePowerMeanOrder { get; init; } = 1;


    /// <summary>
    /// Default constructor.
    /// </summary>
    public NNEvaluatorTorchsharpOptions()
    {
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="qNegativeBlunders"></param>
    /// <param name="qPositiveBlunders"></param>
    /// <param name="fractionUndeblunderedValueHead"></param>
    /// <param name="monitorActivations"></param>
    /// <param name="valueHead1Temperature"></param>
    /// <param name="valueHead2Temperature"></param>
    /// <param name="valueHeadAveragePowerMeanOrder"></param>
    /// <param name="policyTemperatureBase"></param>
    /// <param name="policyTemperatureUncertaintyScalingFactor"></param>
    /// <param name="useAction"></param>
    /// <param name="usePriorState"></param>
    /// <exception cref="ArgumentException"></exception>
    public NNEvaluatorTorchsharpOptions(float qNegativeBlunders = 0, float qPositiveBlunders = 0, 
                                        float fractionUndeblunderedValueHead = 0,
                                        bool monitorActivations = false,
                                        float valueHead1Temperature = 1,
                                        float valueHead2Temperature = 1,
                                        float valueHeadAveragePowerMeanOrder = 1,
                                        float policyTemperatureBase = 1,
                                        float policyTemperatureUncertaintyScalingFactor = 0,
                                        bool useAction = false,
                                        bool usePriorState = false,
                                        float value1UncertaintyTemperatureScalingFactor = 0,
                                        float value2UncertaintyTemperatureScalingFactor = 0)
    {
      if (valueHead1Temperature <= 0 || valueHead2Temperature <= 0)
      {
        throw new ArgumentException("Temperature must be strictly positive.");
      }

      QNegativeBlunders = qNegativeBlunders;
      QPositiveBlunders = qPositiveBlunders;
      FractionValueHead2 = fractionUndeblunderedValueHead;
      MonitorActivations = monitorActivations;
      ValueHead1Temperature = valueHead1Temperature;      
      ValueHead2Temperature = valueHead2Temperature;
      ValueHeadAveragePowerMeanOrder = valueHeadAveragePowerMeanOrder;
      PolicyTemperature = policyTemperatureBase;
      PolicyUncertaintyTemperatureScalingFactor = policyTemperatureUncertaintyScalingFactor;
      UseAction = useAction;
      UsePriorState = usePriorState;
      Value1UncertaintyTemperatureScalingFactor = value1UncertaintyTemperatureScalingFactor;
      Value2UncertaintyTemperatureScalingFactor = value2UncertaintyTemperatureScalingFactor;
    }

    /// <summary>
    /// Short string representation of the options.
    /// </summary>
    public string ShortStr => (UseAction ? "A" : "") + (UsePriorState ? "S" : "")
                           + $"NB: {QNegativeBlunders,4:F2}  PB: {QPositiveBlunders,4:F2} V2: {FractionValueHead2,4:F2} "
                           + $"T: {PolicyTemperature,4:F2}  TS: {PolicyUncertaintyTemperatureScalingFactor,4:F2} ";

  }

}
