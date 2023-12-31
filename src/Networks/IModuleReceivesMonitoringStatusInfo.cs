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

#endregion

namespace CeresTrain.Networks
{
  /// <summary>
  /// Interface which allows a module to receive information about the current monitoring status.
  /// </summary>
  public interface IModuleReceivesMonitoringStatusInfo
  {
    public bool MonitoringCurrentInvocation { get; set; }
  }
}
