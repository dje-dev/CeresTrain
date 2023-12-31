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
using System.Collections.Generic;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Defines a series of learning rate scaling factors
  /// (relative to the base learning rate)
  /// which take effect at a series of successively larger fractions of training completion.
  /// 
  /// Example:
  ///   [(0.01f, 0.1f), (0.7f, 1f), (0.8f, 0.4f)]
  /// indicates to use 0.1 * base learning rate for the first 1% of training,
  /// then 1 * base learning rate up to 70% of training completion,
  /// finally 0.4 * base learning rate for the last 20% of training.
  /// </summary>
  public readonly record struct LRScheduleByFractionComplete
  {
    /// <summary>
    /// Optional enumerable of learning rate multipliers to be applied
    /// at successively larger fractions of training completion.
    /// </summary>
    /// 
    public readonly IList<(float fractionComplete, float lrMultiplier)> Multipliers;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="multipliers"></param>
    public LRScheduleByFractionComplete(IList<(float fractionComplete, float lrMultiplier)> multipliers)
    {
      ValidateLearningRateMultipliers(multipliers);
      Multipliers = multipliers;
    }

    /// <summary>
    /// Implicit conversion from a tuple array to a LearningRateScheduleByFractionComplete.
    /// </summary>
    /// <param name="multipliers"></param>
    public static implicit operator LRScheduleByFractionComplete((float fractionComplete, float lrMultiplier)[] multipliers)
    {
      return new LRScheduleByFractionComplete(multipliers);
    }


    /// <summary>
    /// Return string representation.
    /// </summary>
    /// <returns></returns>
    public override string ToString() => Multipliers.ToString();


    #region Helper methods

    private readonly void ValidateLearningRateMultipliers(IEnumerable<(float fractionComplete, float lrMultiplier)> learningRateMultipliers)
    {
      float previousFractionComplete = -1f;

      foreach ((float fractionComplete, float lrMultiplier) in learningRateMultipliers)
      {
        if (fractionComplete <= previousFractionComplete)
        {
          throw new ArgumentException("fractionComplete values must be in increasing order.");
        }

        if (fractionComplete < 0 || fractionComplete > 1)
        {
          throw new ArgumentOutOfRangeException(nameof(fractionComplete), "fractionComplete must be between 0 and 1.");
        }

        if (lrMultiplier < 0)
        {
          throw new ArgumentOutOfRangeException(nameof(lrMultiplier), "lrMultiplier cannot be negative.");
        }

        previousFractionComplete = fractionComplete;
      }
    }

    /// <summary>
    /// Returns the learning rate multiplier to be applied at the specified fraction of training completion.
    /// </summary>
    /// <param name="fractionComplete"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public float MultiplierForFractionComplete(float fractionComplete)
    {
      if (fractionComplete < 0)
      {
        throw new ArgumentOutOfRangeException(nameof(fractionComplete), "fractionComplete must be at least 0.");
      }

      for (int i = 0; i < Multipliers.Count; i++)
      {
        if (Multipliers[i].fractionComplete > fractionComplete)
        {
          return Multipliers[i].lrMultiplier;
        }
      }

      return Multipliers.Count == 1 ? 1 : Multipliers[^1].lrMultiplier;
    }

    #endregion
  }
}
