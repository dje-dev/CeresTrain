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

using TorchSharp;
using TorchSharp.Modules;

#endregion

namespace CeresTrain.Utils.Tensorboard
{
  /// <summary>
  /// Thin wrapper around writer (for training statistics) to Tensorboard.
  /// 
  /// TODO: Generalize to support other writers such as Weights & Biases.
  /// </summary>
  public class TensorboardWriter
  {
    /// <summary>
    /// Underlying SummaryWriter exposed by TorchSharp.
    /// </summary>
    private readonly SummaryWriter writer;

    /// <summary>
    /// Directory to which log file is written.
    /// </summary>
    public string LogDirectory => logSubdirName;

    private readonly string logSubdirName;


    /// <summary>
    /// Constructor to emit files for a training run with specified tag into a specified base directory.
    /// </summary>
    /// <param name="baseDir"></param>
    /// <param name="trainingRunID"></param>
    /// <exception cref="ArgumentException"></exception>
    public TensorboardWriter(string baseDir, string trainingRunID)
    {
      if (string.IsNullOrEmpty(baseDir))
      {
        throw new ArgumentException("Base directory cannot be null or empty.", nameof(baseDir));
      }

      if (string.IsNullOrEmpty(trainingRunID))
      {
        throw new ArgumentException("Tag cannot be null or empty.", nameof(trainingRunID));
      }

      // Make sure base directory is created
      Directory.CreateDirectory(baseDir);

      // Find a subdirectory index not already used (to avoid collision of different runs/versions).
      int tagSubIndex = 0;
      while (Directory.Exists(Path.Combine(baseDir, trainingRunID, tagSubIndex.ToString())))
      {
        tagSubIndex++;
      }

      // Create subdirectory.
      logSubdirName = Path.Combine(baseDir, trainingRunID, tagSubIndex.ToString());
      Directory.CreateDirectory(logSubdirName);

      Console.WriteLine("Opening Tensorboard log to " + logSubdirName);
      writer = torch.utils.tensorboard.SummaryWriter(logSubdirName);
    }


    /// <summary>
    /// Adds a specified scalar value to the log.
    /// </summary>
    /// <param name="tag"></param>
    /// <param name="value"></param>
    /// <param name="steps"></param>
    public void AddScalar(string tag, float value, long steps)
    {
      steps = Math.Min(int.MaxValue, steps); // Tensorboard limitation

      long walltime = DateTimeOffset.Now.ToUnixTimeSeconds();
      writer.add_scalar(tag, value, (int)steps, walltime);
    }


    /// <summary>
    /// Adds a set of scalars to the log.
    /// </summary>
    /// <param name="steps"></param>
    /// <param name="scalars"></param>
    public void AddScalars(long steps, params (string, float)[] scalars)
    {
      foreach (var (tag, value) in scalars)
      {
        AddScalar(tag, value, steps);
      }
    }
  }
}
