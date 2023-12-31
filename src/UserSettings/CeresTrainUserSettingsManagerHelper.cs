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

namespace CeresTrain.UserSettings
{
  /// <summary>
  /// Helper methods relating to CeresTrain user settings.
  /// </summary>
  internal static class CeresTrainUserSettingsManagerHelper
  {
    /// <summary>
    /// Prompts user for all settings values and writes them to the settings file.
    /// </summary>
    /// <returns></returns>
    internal static CeresTrainUserSettings PromptForUserSettings()
    {
      CeresTrainUserSettings userSettings = new CeresTrainUserSettings();

      Console.WriteLine();
      userSettings.CeresJSONFileName = GetValidFileName ("  CeresJSONFileName : Full path to Ceres.json                                             : ");
      userSettings.TPGSourceTARDir   = GetValidDirectory("  TPGSourceTARDir   : source directory for TAR files containing LC0 training games        : ");
      userSettings.TPGSourceZSTDir   = GetValidDirectory("  TPGSourceZSTDir   : source directory for ZST files containing packed LC0 training games : ");
      userSettings.OutputsDir        = GetValidDirectory("  OutputsDir        : target directory to receive output files (in subdirectories)        : ", false, true);

      return userSettings;
    }


    /// <summary>
    /// Prompts user for a valid directory.
    /// </summary>
    /// <param name="prompt"></param>
    /// <param name="allowNone"></param>
    /// <param name="createIfNeeded"></param>
    /// <returns></returns>
    static string GetValidDirectory(string prompt, bool allowNone = false, bool createIfNeeded = false)
    {
      string inputDir;

      do
      {
        Console.Write(prompt + " ");
        inputDir = Console.ReadLine();
        if (inputDir == "" && allowNone)
        {
          return null;
        }

        if (!Directory.Exists(inputDir))
        {
          if (createIfNeeded)
          {
            Directory.CreateDirectory(inputDir);
          }
          else
          {
            Console.WriteLine("Directory does not exist. Please try again.");
          }
        }
      }
      while (!Directory.Exists(inputDir));

      return inputDir;
    }


    /// <summary>
    /// Prompts user for a valid file name.
    /// </summary>
    /// <param name="prompt"></param>
    /// <param name="allowNone"></param>
    /// <returns></returns>
    static string GetValidFileName(string prompt, bool allowNone = false)
    {
      string inputFN;

      do
      {
        Console.Write(prompt + " ");
        inputFN = Console.ReadLine();
        if (inputFN == "" && allowNone)
        {
          return null;
        }

        if (!File.Exists(inputFN))
        {
          Console.WriteLine("File does not exist. Please try again.");
        }
      }
      while (!File.Exists(inputFN));

      return inputFN;
    }

  }

}

