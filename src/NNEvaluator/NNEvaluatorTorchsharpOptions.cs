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
    public readonly float QNegativeBlunders;
    public readonly float QPositiveBlunders;
    public readonly float FractionValueHead2;
    public readonly bool MonitorActivations;
    public readonly float ValueHead1Temperature;
    public readonly float ValueHead2Temperature;

    public NNEvaluatorTorchsharpOptions(float qNegativeBlunders, float qPositiveBlunders, 
                                        float fractionUndeblunderedValueHead,
                                        bool monitorActivations = false,
                                        float valueHead1Temperature = 1,
                                        float valueHead2Temperature = 1)
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
    }
  }

}
