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

using System.Collections.Concurrent;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using CeresTrain.TPG;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  /// <summary>
  /// Interface implemented by classes which produce TPGRecord[] batches in the background,
  /// typically for use by a Pytorch Dataset implementation.
  /// 
  /// Upon Start method being called, background threads(s) should continually try to populate
  /// the PendingBatchQueue whenever it is less full than the requested maxQueueLength.
  /// </summary>
  public interface ITPGBatchGenerator
  {
    /// <summary>
    /// Begins generation of batches in the background and filling of queue of pending batches.
    /// </summary>
    /// <param name="maxQueueLength"></param>
    void Start(int maxQueueLength);

    /// <summary>
    /// Accessor to the underlying queue of pending batches generated.
    /// </summary>
    ConcurrentQueue<TPGRecord[]> PendingBatchQueue { get; }

    /// <summary>
    /// Starts shutdown of background batch generation.
    /// </summary>
    void Shutdown();
  }
}

