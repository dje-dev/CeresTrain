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
using System.Text.Json.Serialization;

#endregion

namespace CeresTrain.Utils.Tensorboard
{
  /// <summary>
  /// JSON representation of a single scalar event in a Tensorboard event file.
  /// </summary>
  public class TensorboardScalarEvent
  {
    [JsonPropertyName("wall_time")]
    public double WallTimeRaw { get; set; }

    [JsonPropertyName("step")]
    public long Step { get; set; }

    [JsonPropertyName("value")]
    public double Value { get; set; }

    public double Avg { get; set; }


    public DateTime WallDateTime => DateTimeOffset.FromUnixTimeMilliseconds((long)(WallTimeRaw * 1000)).DateTime.ToLocalTime();
  }

}