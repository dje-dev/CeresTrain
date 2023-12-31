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

using Ceres.Base.OperatingSystem;
using Ceres.Base.Misc;
using Ceres.Chess.UserSettings;

using CeresTrain.UserSettings;
using ManagedCuda;
using TorchSharp;
using static TorchSharp.torch;

#endregion

namespace CeresTrain.TrainCommands
{
  internal static class CeresTrainInitialization
  {
    /// <summary>
    /// Verifies that the system prerequisites for running the example code are met.
    /// </summary>
    private static void CheckPrerequisites()
    {
      string tbPath = CeresUserSettingsManager.Settings.SyzygyPath;
      if (tbPath == null || !Path.Exists(tbPath))
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Ceres.json must contain SyzygyPath entry referencing a valid directory, got: " + tbPath);
      }
    }

    static bool haveInitialized = false;
    static readonly object initializingLock = new object();

    /// <summary>
    /// One time start initialization used to prepare the environment for running CeresTrain commands.
    /// </summary>
    public static void InitializeCeresTrainEnvironment()
    {
      if (!haveInitialized)
      {
        lock (initializingLock)
        {
          if (!haveInitialized)
          {
            DoInitialization();
            CheckPrerequisites();
          }
        }
      }
    }

    private static void DoInitialization()
    {
      Console.WriteLine("Initializing CeresTrain environment.");

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

    }


    /// <summary>
    /// Prepares execution environment to begin using a specified device,
    /// performing any possible configuration steps required before switching to this device.
    /// </summary>
    /// <param name="deviceType"></param>
    /// <param name="deviceID"></param>
    public static void PrepareToUseDevice(Device device)
    {
      if (device.type == DeviceType.CUDA)
      {
        const int MIN_CUDA_CAPABILITY_FOR_MATH_SDP = 8; // failed on 2070 (SM7)
        bool okToUsePytorchMathSDP = CudaContext.GetDeviceInfo(device.index).ComputeCapability.Major >= MIN_CUDA_CAPABILITY_FOR_MATH_SDP;
        torch.backends.cuda.enable_math_sdp(okToUsePytorchMathSDP);
      }
    }
  }
}
