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

using Ceres.APIExamples;
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.Data.Nets;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.UserSettings;

using CeresTrain.Examples;
using CeresTrain.TrainCommands;
using CeresTrain.UserSettings;

#endregion

namespace CeresTrain
{
  public static class CeresTrainLauncher
  {
    const string BANNER = @"
|=========================================================|
| CeresTrain - Ceres Neural Network Train/Test Library    |
|                                                         |
| (c)2023 - David Elliott and the CeresTrain Authors      |
|=========================================================|
";

    static void OutputBanner()
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Magenta, "\r\n" + BANNER + "\r\n");
    }


    public static void Main(string[] args)
    {
      OutputBanner();

      // Enable TensorCores. Ideally would do this via TorchSharp, but not currently supported:
      //   torch.backends.cuda.matmul.allow_tf32 = True
      //   torch.backends.cudnn.allow_tf32 = True
      // Instead do via an enviroment variable.
      if (SoftwareManager.IsLinux && Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE") != "1")
      {
        Console.WriteLine("On Linux it is suggested to set environment variable before launch: NVIDIA_TF32_OVERRIDE=1 (if duing CUDA operations in process)");
      }
      else
      {
        Environment.SetEnvironmentVariable("NVIDIA_TF32_OVERRIDE", "1");
      }

      string ceresJSONPath = CeresTrainUserSettingsManager.Settings.CeresJSONFileName;
      Console.WriteLine("Loading " + ceresJSONPath);
      CeresUserSettingsManager.LoadFromFile(ceresJSONPath);

      // ESSENTIAL!
      //   - it seems the torch falls back to CPU for some operations, and starts too many threads.
      //   - if not set, uses default of NumProcessor, CPU goes to 100% as KMP_LAUNCH_WORKER continually starts new threads
      //   - so we set to a small number (but not too small, since some parallelism does seem to improve speed slightly)
      // TODO:CLEANUP
      Environment.SetEnvironmentVariable("OMP_NUM_THREADS", "4");

      // Load the JSON file with configured hosts, if it exists.
      string hostsConfigFileName = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "CeresTrainHosts.json");
      if (File.Exists(hostsConfigFileName))
      {
        Console.WriteLine("Loading " + hostsConfigFileName);
        CeresTrainHostConfig.SetRegisteredHostConfigsFromFile(hostsConfigFileName);
      }
      else
      {
        Console.WriteLine("No hosts config file found at " + hostsConfigFileName);
      }

      CeresTrainCommandLauncher.LaunchProcessCommandLine(args);
    }
  }

}