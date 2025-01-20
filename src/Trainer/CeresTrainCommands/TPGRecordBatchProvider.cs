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
using System.Collections.Generic;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using System.Collections;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Provides batches of TPG records for training.
  /// </summary>
  public class TPGRecordBatchProvider : IEnumerable<TPGRecord[]>
  {
    /// <summary>
    /// Underlying list of TPG records.
    /// </summary>
    public readonly IList<TPGRecord> Records;

    /// <summary>
    /// Number of records in each batch.
    /// </summary>
    public readonly int BatchSize;

    /// <summary>
    /// Total number of batches to provide.
    /// </summary>
    public readonly int TotalBatches;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="records"></param>
    /// <param name="batchSize"></param>
    /// <param name="totalBatches"></param>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public TPGRecordBatchProvider(IList<TPGRecord> records, int batchSize, int totalBatches = int.MaxValue)
    {
      if (records == null || records.Count == 0)
      {
        throw new ArgumentException("Records list cannot be null or empty.", nameof(records));
      }

      if (batchSize <= 0)
      {
        throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be greater than zero.");
      }

      if (totalBatches <= 0)
      {
        throw new ArgumentOutOfRangeException(nameof(totalBatches), "Total batches must be greater than zero.");
      }

      Records = records;
      BatchSize = batchSize;
      TotalBatches = totalBatches;
    }


    public IEnumerator<TPGRecord[]> GetEnumerator()
    {
      int recordCount = Records.Count;

      for (int i = 0; i < TotalBatches; i++)
      {
        TPGRecord[] batch = new TPGRecord[BatchSize];
        for (int j = 0; j < BatchSize; j++)
        {
          batch[j] = Records[(i * BatchSize + j) % recordCount];
        }

        yield return batch;
      }
    }


    IEnumerator IEnumerable.GetEnumerator()
    {
      return GetEnumerator();
    }
  }
}
