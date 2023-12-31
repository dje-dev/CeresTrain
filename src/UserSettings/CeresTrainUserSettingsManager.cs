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
using System.Text.Json;

using Ceres.Base.Misc;

#endregion

namespace CeresTrain.UserSettings
{
  /// <summary>
  /// Manages the reading and writing of CeresTrain user settings files.
  /// </summary>
  public static class CeresTrainUserSettingsManager
  {
    static CeresTrainUserSettings settings = null;

    const string BASE_FN = "CeresTrain.json";


    /// <summary>
    /// Default name of file to which Ceres training user settings are serialized (in JSON).
    /// </summary>
    public static string CeresTrainConfigFileName
    {
      get
      {
        if (File.Exists(BASE_FN))
        {
          // Use file if found in current directory.
          return BASE_FN;
        }
        else
        {
          // Otherwise use file in user's home directory.
          return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), BASE_FN);
        }
      }
    }


    /// <summary>
    /// Current CeresTrain user settings.
    /// </summary>
    public static CeresTrainUserSettings Settings
    {
      get
      {
        if (settings == null)
        {
          if (File.Exists(CeresTrainConfigFileName))
          {
            Load();
          }
          else
          {
            Console.WriteLine($"No CeresTrain user settings file found at {CeresTrainConfigFileName}. Prompting for values:");
            settings = CeresTrainUserSettingsManagerHelper.PromptForUserSettings();
            SaveToFile(CeresTrainConfigFileName);

            Console.WriteLine($"Generated CeresTrain.json file at {CeresTrainConfigFileName} with the following contents:");
            Console.WriteLine(File.ReadAllText(CeresTrainConfigFileName));
          }

          Console.WriteLine();
        }

        return settings;
      }
    }


    /// <summary>
    /// Reads settings from the settings file.
    /// </summary>
    static void Load() => LoadFromFile(CeresTrainConfigFileName);


    /// <summary>
    /// Writes current settings to the settings file.
    /// </summary>
    public static void Save() => SaveToFile(CeresTrainConfigFileName);


    /// <summary>
    /// Reads settings from specified settings file.
    /// </summary>
    static void LoadFromFile(string settingsFileName)
    {
      if (!File.Exists(settingsFileName))
      {
        throw new ArgumentException($"No such file: {settingsFileName}");
      }

      string jsonString = File.ReadAllText(settingsFileName);
      JsonSerializerOptions options = new JsonSerializerOptions() { AllowTrailingCommas = true };
      settings = JsonSerializer.Deserialize<CeresTrainUserSettings>(jsonString, options);

      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"CeresTrain user settings loaded from file {settingsFileName}");
    }


    /// <summary>
    /// Writes current settings to specified settings file.
    /// </summary>
    static void SaveToFile(string settingsFileName)
    {
      string ceresConfigJSON = JsonSerializer.Serialize(settings, new JsonSerializerOptions() { WriteIndented = true });
      File.WriteAllText(settingsFileName, ceresConfigJSON);
    }
  }

}
