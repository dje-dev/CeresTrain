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

#endregion

namespace CeresTrain.NNIntrospection
{
  /// <summary>
  /// Record of set of statistics for a parameter in a neural network.
  /// </summary>
  public readonly record struct ParameterStat
  {
    public readonly string Name;
    public readonly float Min;
    public readonly float Max;
    public readonly float Avg;
    public readonly float NormL2;
    public readonly float StdDev;
    public readonly float Kurtosis;

    public ParameterStat(string name, float min, float max, float normL2, float avg, float stdDev, float kurtosis)
    {
      Name = name ?? throw new ArgumentNullException(nameof(name));
      Min = min;
      Max = max;
      NormL2 = normL2;
      Avg = avg;
      StdDev = stdDev;
      Kurtosis = kurtosis;
    }

    public override string ToString()
    {
      return $"<ParameterStat {Name} Min={Min:F3} Max={Max:F3} L2Norm={NormL2:F3} Avg={Avg:F3} StdDev={StdDev:F3} Kurtosis={Kurtosis:F3}>";
    }

    public void Dump(TextWriter writer)
    {
      writer.WriteLine($"{Name,-55}   [{Min,8:F3} {Max,8:F3} ]   {NormL2,8:F3}  {Avg,8:F3}  +/-{StdDev,8:F3}  K={Kurtosis,8:F3}");
    }
  }

}
