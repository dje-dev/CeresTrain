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
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.Linq;

using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;
using CeresTrain.TPG;
using System.Threading;
using System.Diagnostics;
using Ceres.Chess.NNEvaluators.Ceres.TPG;


#endregion

namespace CeresTrain.TPGDatasets
{
  /// <summary>
  /// Subset of Dataset to provide raw training data sourced from TPG files,
  /// for a subset of files in specified directory matching a specified modulo index.
  /// </summary>
  internal sealed unsafe class TorchDatasetFromTPGWorker : Dataset
  {
    /// <summary>
    /// Directory containing TPG files.
    /// </summary>
    string SourceFileDirectory;

    /// <summary>
    /// Fraction of value target to be derived from Q (search results) rather than game outcome.
    /// </summary>
    public float FractionQ { init; get; }

    /// <summary>
    /// Device onto which tensors are to be loaded.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// Data type of tensors to be created.
    /// </summary>
    public ScalarType DataType { get; }

    /// <summary>
    /// Size of batches to be created.
    /// </summary>
    public int BatchSize { get; }


    TPGBatchToTensorDictConverter dictConverter;

    IEnumerator<(TPGRecord[], bool)> overrideRecordEnumerator;

    int numBatchesRequested;
    int totalNumReaders;
    int indexThisReader;

    ConcurrentQueue<(Dictionary<string, Tensor>, TPGRecord[])> bagPendingDicts = new();



    /// <summary>
    /// Number of batches that can be generated (unlimited due to reusing files as needed).
    /// </summary>
    public override long Count => int.MaxValue;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="tpgDirectory"></param>
    /// <param name="totalNumReaders"></param>
    /// <param name="indexThisReader"></param>
    /// <param name="fractionQ"></param>
    /// <param name="device"></param>
    /// <param name="dataType"></param>
    /// <param name="batchSize"></param>
    /// <param name="wdlLabelSmoothing"></param>
    /// <param name="overrideRecordEnumerator"></param>
    /// <exception cref="ArgumentException"></exception>
    public TorchDatasetFromTPGWorker(string tpgDirectory,
                                     int totalNumReaders, int indexThisReader,
                                     float fractionQ,
                                     Device device, ScalarType dataType, int batchSize,
                                     float wdlLabelSmoothing,
                                     IEnumerator<(TPGRecord[], bool)> overrideRecordEnumerator)
    {
      if (totalNumReaders == 0)
      {
        throw new ArgumentException(nameof(totalNumReaders), "is zero");
      }

      if (indexThisReader >= totalNumReaders)
      {
        throw new ArgumentException(nameof(indexThisReader), "too large for specified number of readers");
      }

      dictConverter = new TPGBatchToTensorDictConverter(fractionQ, device, dataType, batchSize, wdlLabelSmoothing);

      FractionQ = fractionQ;
      Device = device;
      DataType = dataType;
      BatchSize = batchSize;
      SourceFileDirectory = tpgDirectory;

      this.totalNumReaders = totalNumReaders;
      this.indexThisReader = indexThisReader;
      this.overrideRecordEnumerator = overrideRecordEnumerator;

      Reset();
      Task.Run(TaskPreloadDicts);
    }


    /// <summary>
    /// Starts or restarts the set of files to be processed.
    /// </summary>
    /// <exception cref="Exception"></exception>
    void Reset()
    {
      // TODO: is some special treatment necessary for when usingTPG is false?  

      bool usingTPG = overrideRecordEnumerator == null;
      if (usingTPG && SourceFileDirectory != null)
      {
        // Get list of files and sort so different workers see files ordered the same
        // (for consistent modulus selection).
        List<string> fileNames = Directory.GetFiles(SourceFileDirectory, "*.zst").ToList();

        // Sort fileNames based on name
        fileNames.Sort((a, b) => string.Compare(a, b, StringComparison.Ordinal));

        availableFiles = new ConcurrentQueue<string>();
        for (int i = 0; i < fileNames.Count; i++)
        {
          // Select exactly the files that match the modulo index for this worker.
          if (i % totalNumReaders == indexThisReader)
          {
            availableFiles.Enqueue(fileNames[i]);
          }
        }

        if (availableFiles.Count == 0)
        {
          throw new Exception("No files available " + SourceFileDirectory + " moduloIndex " + indexThisReader);
        }
      }
    }


    IEnumerator<(TPGRecord[], bool)> currentFileEnumerator = null;
    ConcurrentQueue<string> availableFiles = null;

    /// <summary>
    /// Advances to next file,  restarts if all files have been processed.
    /// </summary>
    /// <exception cref="Exception"></exception>
    void StartNextFile()
    {
      if (overrideRecordEnumerator != null)
      {
        currentFileEnumerator = overrideRecordEnumerator;
      }
      else
      {
        if (!availableFiles.TryDequeue(out string thisFN))
        {
          // Have exhausted set of files, reload them all and start over.
          Reset();

          if (!availableFiles.TryDequeue(out thisFN))
          {
            throw new Exception("Failure restarting processing of TPG files in " + SourceFileDirectory + " moduloIndex " + indexThisReader);
          }
        }

        TPGFileReader reader = new TPGFileReader(thisFN, dictConverter.BatchSize);
        currentFileEnumerator = EnumeratorAddBoolean(reader.GetEnumerator());
      }
    }

    public static IEnumerator<(TPGRecord[], bool)> EnumeratorAddBoolean(IEnumerator<TPGRecord[]> source)
    {
      while (source.MoveNext())
      {
        yield return (source.Current, true);
      }
    }


    /// <summary>
    /// Override of base method to return a batch of tensors.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public override Dictionary<string, Tensor> GetTensor(long index) => GetTensorAndRawRecords(index, DataType).dict;


    /// <summary>
    /// 
    /// </summary>
    /// <param name="index">index of batch to retrieve (ignored)</param>
    /// <param name="dataType">data type of returned batch (currently ignored)</param>
    /// <returns></returns>
    public (Dictionary<string, Tensor> dict, TPGRecord[] rawRecords) GetTensorAndRawRecords(long index, ScalarType dataType)
    {
      Debug.Assert(dataType == DataType);

      numBatchesRequested++;

      // Dequeue a batch of records, looping/waiting if not yet available.
      (Dictionary<string, Tensor> dict, TPGRecord[] rawRecords) ret;
      while (!bagPendingDicts.TryDequeue(out ret))
      {
        Thread.Sleep(10);
      }

      return ret;
    }


    // Use at least 2 to insure overlapping. Larger values require more GPU memory.
    const int NUM_DICTS_PRELOAD = 2; 

    void TaskPreloadDicts()
    {
      while (true)
      {
        (TPGRecord[] theseRecords, bool isPrimary) thisRet = default;
        if (bagPendingDicts.Count >= NUM_DICTS_PRELOAD)
        {
          // Queue already full enough, wait a short while.
          Thread.Sleep(30);
        }
        else
        {
          if (overrideRecordEnumerator != null)
          {
            overrideRecordEnumerator.MoveNext();
            thisRet = overrideRecordEnumerator.Current;
          }
          else
          {
            if (availableFiles != null)
            {
              if (currentFileEnumerator == null)
              {
                StartNextFile();
              }

              while (!currentFileEnumerator.MoveNext())
              {
                StartNextFile();
              }
            }

            thisRet = currentFileEnumerator.Current;
          }

        }

        if (thisRet != default)
        {
          bagPendingDicts.Enqueue((dictConverter.BuildTensorDictFromTPGRecords(thisRet.theseRecords, thisRet.isPrimary), thisRet.theseRecords));
        }
      }
    }


  }
}
