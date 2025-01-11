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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

using Zstandard.Net;
using Ceres.Base.DataType;
using Ceres.Chess.NNEvaluators.Ceres.TPG;

#endregion


namespace CeresTrain.TPG
{
  /// <summary>
  /// TPGReader facilitates efficient sequential enumeration 
  /// of TPG records contained in a compressed (ZStandard ZST) files).
  /// 
  /// A single background thread is launched to continually read ahead 
  /// so enumeration requests can be satisfied immediately.
  /// </summary>
  public class TPGFileReader : IEnumerable<TPGRecord[]>
  {
    /// <summary>
    /// Name of the TPG file being read.
    /// </summary>
    public readonly string TPGFileName;

    /// <summary>
    /// Number of records in each batch.
    /// </summary>
    public readonly int BatchSize;

    /// <summary>
    /// Is set to true to signal that the reader should stop reading.
    /// </summary>
    public bool shouldShutdown = false;

    // Single-pass enumerator
    private IEnumerator<TPGRecord[]> enumerator;

    const int READAHEAD_COUNT = 8;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="tpgFileName"></param>
    /// <param name="batchSize"></param>
    public TPGFileReader(string tpgFileName, int batchSize)
    {
      ArgumentNullException.ThrowIfNull(tpgFileName);
      TPGFileName = tpgFileName;
      BatchSize = batchSize;

      // Create the single-pass enumerator
      enumerator = TPGBatchEnumerator(tpgFileName, batchSize).GetEnumerator();
    }

    /// <summary>
    /// Signals the reader to stop reading.
    /// </summary>
    public void Shutdown() => shouldShutdown = true;

    /// <summary>
    /// Returns the next batch of records from the TPG file.
    /// </summary>
    public TPGRecord[] NextBatch()
    {
      if (!enumerator.MoveNext())
      {
        Console.WriteLine($"No more data in TPG file {TPGFileName}");
        return null;
      }
      return enumerator.Current;
    }

    /// <summary>
    /// The main enumerator logic that yields batches (TPGRecord[]).
    /// Runs a background thread to fill a queue of batches.
    /// </summary>
    IEnumerable<TPGRecord[]> TPGBatchEnumerator(string fileName, int batchSize)
    {
      DateTime startTime = DateTime.Now;

      FileStream es = new FileStream(fileName, FileMode.Open, FileAccess.Read);
      Stream decompressionStream = new ZstandardStream(es, CompressionMode.Decompress);
      int totNumRead = 0;
      bool allDone = false;

      ConcurrentQueue<TPGRecord[]> pendingBatches = new();

      void BackgroundFillWithExceptionHandler()
      {
        try
        {
          BackgroundFillWorker();
        }
        catch (Exception e)
        {
          Console.WriteLine("Exception in BackgroundFillWorker decompressing/processing TPG records from " + fileName, e);

          // Abruptly shutdown so this messages does not get overwritten by other output which may refresh/repaint Console.
          System.Environment.Exit(3);
        }
      }

      void BackgroundFillWorker()
      {
        TPGRecord[] readBuffer = new TPGRecord[batchSize];
        byte[] bufferBackgroundThread = new byte[Marshal.SizeOf<TPGRecord>() * batchSize];

        while (!allDone && !shouldShutdown)
        {
          if (pendingBatches.Count < READAHEAD_COUNT)
          {
            int numRead = StreamUtils.ReadFromStream(
                decompressionStream,
                bufferBackgroundThread,
                ref readBuffer,
                batchSize
            );

            totNumRead += numRead;
            if (numRead != batchSize)
            {
              allDone = true;
            }
            else
            {
              // Copy so we don't overwrite readBuffer
              TPGRecord[] copy = new TPGRecord[batchSize];
              Array.Copy(readBuffer, copy, batchSize);
              pendingBatches.Enqueue(copy);
            }
          }
          else
          {
            Thread.Sleep(30);
          }
        }
      }

      // Launch the background fill
      Task.Run(BackgroundFillWithExceptionHandler);

      using (decompressionStream)
      {
        while (true)
        {
          TPGRecord[] ret = null;
          while (!pendingBatches.TryDequeue(out ret))
          {
            if (allDone)
            {
              yield break;
            }
            else
            {
              Thread.Sleep(30);
            }
          }

          yield return ret;
        }
      }
    }


    #region IEnumerable<TPGRecord[]> 

    /// <summary>
    /// The generic GetEnumerator() is required by IEnumerable<T>.
    /// This yields from our single-pass enumerator.
    /// </summary>
    public IEnumerator<TPGRecord[]> GetEnumerator()
    {
      // Single pass: once enumerator is exhausted, 
      // subsequent calls return an empty sequence.
      while (enumerator.MoveNext())
      {
        yield return enumerator.Current;
      }
    }

    /// <summary>
    /// Returns an IEnumerable of individual TPGRecords by flattening the batch enumerator.
    /// </summary>
    public IEnumerable<TPGRecord> Records
    {
      get
      {
        foreach (TPGRecord[] batch in this)
        {
          if (batch == null)
          {
            yield break;
          }

          foreach (TPGRecord record in batch)
          {
            yield return record;
          }
        }
      }
    }


    /// <summary>
    /// Non-generic version of GetEnumerator(), required by IEnumerable.
    /// </summary>
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    #endregion
  }
}
