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

using System.IO;

#endregion

namespace CeresTrain.UserSettings
{
  /// <summary>
  /// Set of configurable user settings for CeresTrain
  /// which define various system configuration values (e.g. directories)
  /// and preferences.
  /// </summary>
  public record CeresTrainUserSettings
  {
    /// <summary>
    /// Directory
    /// </summary>
    public string CeresJSONFileName { get; set; } = ".";

    /// <summary>
    /// Directory from which TAR files are read to be used with TPG generation.
    /// </summary>
    public string TPGSourceTARDir { get; set; } = ".";

    /// <summary>
    /// Directory from which ZST files are read to be used with TPG generation (packed TAR files).
    /// </summary>
    public string TPGSourceZSTDir { get; set; } = ".";

    /// <summary>
    /// Directory to various types of output files are written (in subdirectories automatically created).
    /// </summary>
    public string OutputsDir { get; set; } = ".";


    /// <summary>
    /// Directory to which TPG files are written.
    /// </summary>
    public string OutputTPGDir => MakeDir(OutputsDir, "tpg");

    /// <summary>
    /// Directory to which networks being tested are cached during testing.
    /// </summary>
    public string OutputNetsDir => MakeDir(OutputsDir, "nets");

    /// <summary>
    /// Directory to which TPG files are written.
    /// </summary>
    public string OutputLogsDir => MakeDir(OutputsDir, "logs");


    #region Helper methods

    /// <summary>
    /// Create subdirectory name and create if not already extant.
    /// </summary>
    /// <param name="baseDir"></param>
    /// <param name="subdirName"></param>
    /// <returns></returns>
    string MakeDir(string baseDir, string subdirName)
    {
      string dirName = Path.Combine(baseDir, subdirName);
      if (!Directory.Exists(dirName))
      {
        // Create dir
        Directory.CreateDirectory(dirName);
      }

      return dirName;
    }

    #endregion

  }
}
