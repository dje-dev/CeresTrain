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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

using Zstandard.Net;
using Ceres.Base.DataType;


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
  public class TPGFileReader
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


    IEnumerator<TPGRecord[]> enumerator;

    // Alternate between multiple readahead buffers
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

      enumerator = TPGBatchEnumerator(tpgFileName, batchSize).GetEnumerator();
    }


    /// <summary>
    /// Enumerator that returns batches of records from the TPG file.
    /// </summary>
    public IEnumerator<TPGRecord[]> Enumerator { get => enumerator; }


    /// <summary>
    /// Returns IEnumerable that enumerates batches of records from the TPG file.
    /// </summary>
    public IEnumerable<TPGRecord[]> Enumerable
    {
      get
      {
        while (enumerator.MoveNext())
        {
          yield return enumerator.Current;
        }
      }
    }



    /// <summary>
    /// Signals the reader to stop reading.
    /// </summary>
    public void Shutdown() => shouldShutdown = true;


    /// <summary>
    /// Returns the next batch of records from the TPG file.
    /// </summary>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public TPGRecord[] NextBatch()
    {
      if (!enumerator.MoveNext())
      {
        throw new Exception($"Ran out of data in TPG file {TPGFileName}");
      }

      return enumerator.Current;
    }


    /// <summary>
    /// Worker method that reads batches of records from the TPG file,
    /// including a background thread that reads ahead.
    /// </summary>
    /// <param name="fileName"></param>
    /// <param name="batchSize"></param>
    /// <returns></returns>
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
            int numRead = StreamUtils.ReadFromStream(decompressionStream, bufferBackgroundThread, ref readBuffer, batchSize);

            totNumRead += numRead;
            if (numRead != batchSize)
            {
              allDone = true;
            }
            else
            {
              // Make a copy and enqueue.
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

      Task.Run(BackgroundFillWithExceptionHandler);

      decompressionStream = new ZstandardStream(es, CompressionMode.Decompress);
      using (decompressionStream)
      {
        while (true)
        {
          TPGRecord[] ret = default;
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

  }
}
