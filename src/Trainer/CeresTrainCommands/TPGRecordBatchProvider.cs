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
using System.Collections;
using System.Collections.Generic;

using Ceres.Chess.NNEvaluators.Ceres.TPG;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Provides batches of TPG records for training.
  /// </summary>
  public class TPGRecordBatchProvider : IEnumerable<(TPGRecord[] records, bool isSet1)>
  {
    /// <summary>
    /// Underlying list of TPG records, primary set.
    /// </summary>
    public readonly IList<TPGRecord> Set1Records;

    /// <summary>
    /// Underlying list of TPG records, secondary set.
    /// </summary>
    public readonly IList<TPGRecord> Set2Records;

    /// <summary>
    /// Number of records in each batch.
    /// </summary>
    public readonly int BatchSize;

    /// <summary>
    /// Total number of batches to provide.
    /// </summary>
    public readonly int TotalBatches;

    /// <summary>
    /// How many set 1 batches to produce before yielding one set 2 batch.
    /// </summary>
    public readonly int NumSet1ForEachSet2Record;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="set1Records">First set of TPG records</param>
    /// <param name="set2Records">Second set of TPG records</param>
    /// <param name="batchSize">Batch size</param>
    /// <param name="numSet1ForEachSet2Record">
    /// Number of consecutive primary batches before taking one secondary batch
    /// </param>
    /// <param name="totalBatches">Total batches to provide</param>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public TPGRecordBatchProvider(IList<TPGRecord> set1Records,
                                  IList<TPGRecord> set2Records,
                                  int batchSize,
                                  int numSet1ForEachSet2Record,
                                  int totalBatches = int.MaxValue)
    {
      if (numSet1ForEachSet2Record > 0 && (set1Records == null || set1Records.Count == 0))
      {
        throw new ArgumentException("primaryRecords cannot be null or empty.", nameof(set1Records));
      }

      if (set2Records == null || set2Records.Count == 0)
      {
        throw new ArgumentException("secondaryRecords cannot be null or empty.", nameof(set2Records));
      }

      if (batchSize <= 0)
      {
        throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be greater than zero.");
      }
 
      if (totalBatches <= 0)
      {
        throw new ArgumentOutOfRangeException(nameof(totalBatches), "Total batches must be greater than zero.");
      }

      Set1Records = set1Records;
      Set2Records = set2Records;
      BatchSize = batchSize;
      NumSet1ForEachSet2Record = numSet1ForEachSet2Record;
      TotalBatches = totalBatches;
    }


    public IEnumerator<(TPGRecord[] records, bool isSet1)> GetEnumerator()
    {
      int primaryCount = Set1Records.Count;
      int secondaryCount = Set2Records.Count;

      for (int i = 0; i < TotalBatches; i++)
      {
        bool useSet1 = (i % (NumSet1ForEachSet2Record + 1)) < NumSet1ForEachSet2Record;
        IList<TPGRecord> sourceRecords = useSet1 ? Set1Records : Set2Records;
        int recordCount = useSet1 ? primaryCount : secondaryCount;

        TPGRecord[] batch = new TPGRecord[BatchSize];
        for (int j = 0; j < BatchSize; j++)
        {
          batch[j] = sourceRecords[(i * BatchSize + j) % recordCount];
        }

        yield return (batch, useSet1);
      }
    }


    IEnumerator IEnumerable.GetEnumerator()
    {
      return GetEnumerator();
    }
  }
}
