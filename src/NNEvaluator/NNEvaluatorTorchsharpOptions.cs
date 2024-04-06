#region Using directives

using System;

#endregion

namespace CeresTrain.NNEvaluators
{
  /// <summary>
  /// Set of options for NNEvaluatorTorchsharp.
  /// </summary>
  public record class NNEvaluatorTorchsharpOptions
  {
    /// <summary>
    /// If the action head should be used.
    /// </summary>
    public readonly bool HasAction = true;

    /// <summary>
    /// Assumed magnitude (Q units) of adverse blunders that will follow in the game.
    /// </summary>
    public readonly float QNegativeBlunders;

    /// <summary>
    /// Assumed magnitude (Q units) of favorable blunders that will follow in the game.
    /// </summary>
    public readonly float QPositiveBlunders;

    /// <summary>
    /// Fraction of the value head 2 that is used to blend into the primary value.
    /// </summary>
    public readonly float FractionValueHead2;

    /// <summary>
    /// If true, monitor activations of the neural network. 
    /// </summary>
    public readonly bool MonitorActivations;

    /// <summary>
    /// Temperature for the value head 1.
    /// </summary>
    public readonly float ValueHead1Temperature;

    /// <summary>
    /// Temperature for the value head 2.
    /// </summary>
    public readonly float ValueHead2Temperature;

    /// <summary>
    /// The order of the power mean used to combine the value heads.
    /// Default value of 1 corresponds to the arithmetic mean (0 for geometric).
    /// </summary>
    public readonly float ValueHeadAveragePowerMeanOrder = 1;

    /// <summary>
    /// If extreme values (near -1 or 1) 
    /// </summary>
    public readonly bool ShrinkExtremes;



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="qNegativeBlunders"></param>
    /// <param name="qPositiveBlunders"></param>
    /// <param name="fractionUndeblunderedValueHead"></param>
    /// <param name="monitorActivations"></param>
    /// <param name="valueHead2Temperature"></param>
    /// <param name="valueHead1Temperature"></param>
    /// <param name="shrinkExtremes"></param>
    /// <exception cref="ArgumentException"></exception>
    public NNEvaluatorTorchsharpOptions(float qNegativeBlunders = 0, float qPositiveBlunders = 0, 
                                        float fractionUndeblunderedValueHead = 0,
                                        bool monitorActivations = false,
                                        float valueHead1Temperature = 1,
                                        float valueHead2Temperature = 1,
                                        float valueHeadAveragePowerMeanOrder = 1,
                                        bool shrinkExtremes = false)
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
      ShrinkExtremes = shrinkExtremes;
    }
  }

}
