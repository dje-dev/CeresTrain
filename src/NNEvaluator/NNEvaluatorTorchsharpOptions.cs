#region Using directives

using System;

#endregion

namespace CeresTrain.NNEvaluators
{
  /// <summary>
  /// Set of options for NNEvaluatorTorchsharp.
  /// </summary>
  public readonly record struct NNEvaluatorTorchsharpOptions
  {
    /// <summary>
    /// If the prior state information should be used.
    /// </summary>
    public readonly bool UsePriorState { get; init; } = false;

    /// <summary>
    /// If the action head should be used.
    /// </summary>
    public readonly bool UseAction { get; init; } = false;

    /// <summary>
    /// Assumed magnitude (Q units) of adverse blunders that will follow in the game.
    /// </summary>
    public readonly float QNegativeBlunders { get; init; } = 0;

    /// <summary>
    /// Assumed magnitude (Q units) of favorable blunders that will follow in the game.
    /// </summary>
    public readonly float QPositiveBlunders { get; init; } = 0;

    /// <summary>
    /// Fraction of the value head 2 that is used to blend into the primary value.
    /// </summary>
    public readonly float FractionValueHead2 { get; init; } = 0;

    /// <summary>
    /// If true, monitor activations of the neural network. 
    /// </summary>
    public readonly bool MonitorActivations { get; init; } = false;


    /// <summary>
    /// Temperature for the value head 2.
    /// </summary>
    public readonly float ValueHead1Temperature
    {
      get => valueHead1Temperature;
      init => valueHead1Temperature = value <= 0 ? throw new ArgumentException("Temperature must be strictly positive.") : value;
    }

    /// <summary>
    /// Temperature for the value head 2.
    /// </summary>
    public readonly float ValueHead2Temperature
    {
      get => valueHead2Temperature;
      init => valueHead2Temperature = value <= 0 ? throw new ArgumentException("Temperature must be strictly positive.") : value;
    }


    /// <summary>
    /// The order of the power mean used to combine the value heads.
    /// Default value of 1 corresponds to the arithmetic mean (0 for geometric).
    /// </summary>
    public readonly float ValueHeadAveragePowerMeanOrder { get; init; } = 1;

    /// <summary>
    /// Optional scaling factor that determines the amount by which 
    /// the policy temperature is scaled based on position-specific policy uncertainty.
    /// </summary>
    public readonly float PolicyTemperatureScalingFactor { get; init; } = 0;

    /// <summary>
    /// Base policy temperature to apply.
    /// </summary>
    public readonly float PolicyTemperatureBase { get; init; } = 1.0f;

    private readonly float valueHead2Temperature = 1;
    private readonly float valueHead1Temperature = 1;


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
    /// <param name="policyTemperatureScalingFactor"></param>
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
                                        float policyTemperatureScalingFactor = 0,
                                        bool useAction = false,
                                        bool usePriorState = false)
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
      PolicyTemperatureBase = policyTemperatureBase;
      PolicyTemperatureScalingFactor = policyTemperatureScalingFactor;
      UseAction = useAction;
      UsePriorState = usePriorState;
    }
  }

}
