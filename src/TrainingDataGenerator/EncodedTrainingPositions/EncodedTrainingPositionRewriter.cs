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
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using System.Collections.Generic;

using System.Formats.Tar;
using System.IO.Compression;

using Ceres.Chess.EncodedPositions;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataType;
using Ceres.Base.Misc;


#endregion

namespace CeresTrain.TrainingDataGenerator
{
  /// <summary>
  /// Set of helper methods to rewrite TAR files with training positions
  /// into Ceres formatted TAR files with two changes:
  ///   - instead of chunks containing EncodedTrainingPosition[],
  ///     the are converted to EncodedTrainingPositionCompressed[]
  ///     which greatly reduces size in bytes (due to policy being sparsely encoded).
  ///   - the inner TAR file is compressed with ZSTD (instead of GZIP).
  ///   
  /// The reader class EncodedTrainingPositionReaderTAREngine will automatically
  /// recognize which format any given file uses and always return EncodedTrainingPosition[],
  /// decompressing and decoding if the TAR entries have extensions 
  /// of .zst indicating they are Ceres compressed.
  /// </summary>
  public static partial class EncodedTrainingPositionRewriter
  {
    public const int DEFAULT_COMPRESSION_LEVEL = 8;

    const int MAX_GAMES_PER_BLOCK = 100;
    const int MAX_AVG_POSITIONS_PER_GAME = 300;
    const int MAX_POSITIONS_PER_BLOCK = MAX_GAMES_PER_BLOCK * MAX_AVG_POSITIONS_PER_GAME;

    /// <summary>
    /// Converts multiple TARs in parallel.
    /// </summary>
    /// <param name="sourceDir"></param>
    /// <param name="targetDir"></param>
    /// <param name="filenameFilter"></param>
    /// <param name="writeBlocked"></param>
    /// <param name="compressionLevel"></param>
    /// <param name="maxParallel"></param>
    /// <param name="maxGamesPerTAR"></param>
    public static void ConvertTARDir(string sourceDir, string targetDir,
                                     string filenameFilter,
                                     bool writeBlocked,
                                     int compressionLevel,
                                     int maxParallel,
                                     Predicate<Memory<EncodedTrainingPosition>> acceptGamePredicate = default,
                                     LC0TrainingPosGeneratorFromSingleNNEval extraTrainingPositionGenerator = null,
                                     int maxGamesPerTAR = int.MaxValue)
    {
      DateTime startTime = DateTime.Now;

      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "EncodedTrainingPositionRewriter.ConvertTARDir");  
      Console.WriteLine("Converting training positions from LC0 format to Ceres format.");
      Console.WriteLine("  Source dir: " + sourceDir);
      Console.WriteLine("  Target dir: " + targetDir);

      // Get file list, only include those in TAR format that are not already in Ceres ZST format.
      filenameFilter = filenameFilter ?? "*";
      List<string> files = Directory.EnumerateFiles(sourceDir, filenameFilter).ToList();
      files = files.Where(f => f.ToLower().EndsWith(".tar") &&
                              !f.ToLower().Contains(".zst")).ToList();

      Console.WriteLine("  Number of files to convert : " + files.Count());
      Console.WriteLine("  Game accept predicate?     : " + (acceptGamePredicate == null ? "N" : "Y"));
      Console.WriteLine("  Compression level          : " + compressionLevel);
      Console.WriteLine("  Maximum number in parallel : " + maxParallel);
      Console.WriteLine();

      long numGamesTotal = 0;
      long numPosWrittenTotal = 0;
      long numGamesSkippedTotal = 0;
      long numSupplementalPosTotal = 0;
      long bytesReadTotal = 0;
      long bytesWrittenTotal = 0; 

      Parallel.ForEach(files, new ParallelOptions() { MaxDegreeOfParallelism = maxParallel },
        file =>
        {
          FileInfo sourceInfo = new FileInfo(file);
          string targetFN = Path.Combine(targetDir, sourceInfo.Name.Replace(".tar", ".zst.tar"));
          if (File.Exists(targetFN) && new FileInfo(targetFN).Length > 0)
          {
            Console.WriteLine("skip (already present) " + targetFN);
            return;
          }

          using (new TimingBlock(file))
          {
            (long numGames, long numGamesSkipped, long numPosWritten, long numSupplementalPos,
            long numBytesRead, long numBytesWritten) stats = default;

            try
            {
              stats = ConvertTAR(file, acceptGamePredicate, targetFN, writeBlocked, extraTrainingPositionGenerator, compressionLevel, maxGamesPerTAR);
              Interlocked.Add(ref numGamesTotal, stats.numGames);
              Interlocked.Add(ref numGamesSkippedTotal, stats.numGamesSkipped);
              Interlocked.Add(ref numPosWrittenTotal, stats.numPosWritten);
              Interlocked.Add(ref numSupplementalPosTotal, stats.numSupplementalPos);
              Interlocked.Add(ref bytesReadTotal, stats.numBytesRead);
              Interlocked.Add(ref bytesWrittenTotal, stats.numBytesWritten);
            }
            catch (Exception e)
            {
              Console.WriteLine("Abort TAR, exception encountered " + file);
              Console.WriteLine(e);
            }
          }
        });

      var elapsedTime = DateTime.Now - startTime;

      Console.WriteLine();
      Console.WriteLine("Number bytes read total                  : " + bytesReadTotal.ToString("N0"));
      Console.WriteLine("Number bytes written total               : " + bytesWrittenTotal.ToString("N0"));
      Console.WriteLine("Compression ratio                        : " + ((double)bytesWrittenTotal/ bytesReadTotal).ToString("F3"));  
      Console.WriteLine("Elapsed seconds                          : " + elapsedTime.TotalSeconds.ToString("N2"));
      Console.WriteLine("TAR processing rate                      : " + (bytesReadTotal / elapsedTime.TotalSeconds / 1e6).ToString("N0") + " MB/s");
      Console.WriteLine();
      Console.WriteLine("Number games processed total             : " + numGamesTotal.ToString("N0"));
      Console.WriteLine("Number games skipped due to Predicate    : " + numGamesSkippedTotal.ToString("N0"));
      Console.WriteLine("Number positions written total           : " + numPosWrittenTotal.ToString("N0"));

      Console.WriteLine("Number supplemental positions generated  : " + numSupplementalPosTotal.ToString("N0"));
      Console.WriteLine();
    }



    /// <summary>
    /// Rewrites a single source TAR (from original LC0 training data) 
    /// into a target TAR with the enhanced compression features described below.
    /// </summary>
    /// <param name="sourceTARName"></param>
    /// <param name="acceptGamePredicate"></param>
    /// <param name="targetZSTName"></param>
    /// <param name="writeBlocked"></param>
    /// <param name="extraTrainingPositionGenerator"></param>
    /// <param name="compressionLevel"></param>
    /// <param name="maxGames"></param>
    /// <returns></returns>
    public static (long numGames, long numGamesSkipped, long numPosWritten, long numSupplementalPos,
                   long bytesRead, long bytesWritten) 
      ConvertTAR(string sourceTARName,
                 Predicate<Memory<EncodedTrainingPosition>> acceptGamePredicate,
                 string targetZSTName,
                 bool writeBlocked,
                 LC0TrainingPosGeneratorFromSingleNNEval extraTrainingPositionGenerator,
                 int compressionLevel = DEFAULT_COMPRESSION_LEVEL,
                 int maxGames = int.MaxValue)
    {
      EncodedTrainingPosition[] encodedTrainingPositions = new EncodedTrainingPosition[MAX_GAMES_PER_BLOCK];

      PositionsMultipleGamesWriter writer = writeBlocked ? new(MAX_GAMES_PER_BLOCK, compressionLevel) : null;

      long posBytesWritten = 0;
      long posBytesRead = new FileInfo(sourceTARName).Length;
      long numGamesWritten = 0;
      long numGamesSkippedDueToAcceptPredicate = 0;

      long numPosWritten = 0;
      long numSupplementalTrainingPositionsGenerated = 0;

      using FileStream sourceStream = File.OpenRead(sourceTARName);
      using TarReader tarReader = new TarReader(sourceStream);

      // Create a new TAR file to write into
      using FileStream targetStream = File.Create(targetZSTName);
      using TarWriter tarWriter = new TarWriter(targetStream);

      int numEntries = 0;
      while (true)
      {
        TarEntry entry = tarReader.GetNextEntry();
        if (entry == null)
        {
          break;
        }

        numEntries++;

#if NOT
        if (numEntries % 1000 == 0)
        {
          Console.WriteLine("  " + numEntries + " " + ((float)posBytesWritten / posBytesRead));
        }
#endif

        // Ignore directories
        if (entry.EntryType == TarEntryType.RegularFile)
        {
          using MemoryStream decompressedData = new MemoryStream();

          // Decompress Zip data
          using Stream entryStream = entry.DataStream;
          if (entryStream == null)
          {
            continue;
          }

          bool isGZ = entry.Name.EndsWith(".gz");
          using Stream decompressedEntryStream = isGZ ? new GZipStream(entryStream, CompressionMode.Decompress)
                                                      : entryStream; // not compressed, just copy as is

          int numPosReadThisGame = 0;
          try
          {
            decompressedEntryStream.CopyTo(decompressedData);

            if (writer == null)
            {
              // Extract from stream, without any sort of unmirroring.
              numPosReadThisGame = ExtractEncodedTrainingPositions(decompressedData, acceptGamePredicate, encodedTrainingPositions, false);
            }
          }
          catch (Exception e)
          {
            Console.WriteLine("FAIL " + entry.Name + " " + entry.EntryType);
            continue;
          }

          if (!isGZ || writer == null)
          {
            if (numPosReadThisGame == 0)
            {
              numGamesSkippedDueToAcceptPredicate++;
            }
            else
            {
              // Recompress (using ZStandard) positions with no change to mirroring state.
              string newFN = WriteToTAR(compressionLevel, ref posBytesWritten,
                                        tarWriter, entry.Name, decompressedData, isGZ, encodedTrainingPositions.AsSpan().Slice(0, numPosReadThisGame));
              File.Delete(newFN);

              if (isGZ)
              {
                numGamesWritten++;
                numPosWritten += numPosReadThisGame;
              }
            }
          }
          else
          {
            // Flush buffer if necessary.
            if (writer.NumGamesInBuffer >= MAX_GAMES_PER_BLOCK)
            {
              writer.WriteBlock(tarWriter);
            }

            numPosReadThisGame = writer.AddGamePositionsFromStream(decompressedData, acceptGamePredicate,
                                                                   extraTrainingPositionGenerator, out bool finalTrainingPosWasGenerated);
            if (numPosReadThisGame == 0)
            {
              numGamesSkippedDueToAcceptPredicate++;
            }
            else
            {
              numGamesWritten++;
              numPosWritten += numPosReadThisGame;

              if (finalTrainingPosWasGenerated)
              {
                numSupplementalTrainingPositionsGenerated++;
                numPosWritten++;
              }
            }
          }
        }

        if (numGamesWritten >= maxGames)
        {
          break;
        }
      }

      // Write the final block, if any.
      if (writer != null)
      {
        numGamesWritten += writer.NumGamesInBuffer;
        numPosWritten += writer.NumPosInBuffer;
        writer.WriteBlock(tarWriter);
      }

      posBytesWritten = new FileInfo(targetZSTName).Length;
      tarWriter.Dispose();

      return (numGamesWritten, numGamesSkippedDueToAcceptPredicate, numPosWritten, 
              numSupplementalTrainingPositionsGenerated,
              posBytesRead, posBytesWritten);
    }

    [ThreadStatic]
    static EncodedTrainingPositionCompressed[] compressedPositions;


    private static string WriteToTAR(int compressionLevel,
                                     ref long posBytesWritten,
                                     TarWriter tarWriter, string entryName, MemoryStream decompressedData,
                                     bool isGZ, Span<EncodedTrainingPosition> encodedTrainingPositions)
    {
      string tempFN = Path.GetTempFileName();

      using (FileStream compressedFile = new(tempFN, FileMode.Create))
      {
        if (compressedPositions == null)
        {
          compressedPositions = new EncodedTrainingPositionCompressed[MAX_POSITIONS_PER_BLOCK];
        }

        if (isGZ)
        {
          // Convert the position into compressed form.
          int numPositions = encodedTrainingPositions.Length;
          EncodedTrainingPositionCompressedConverter.Compress(encodedTrainingPositions, compressedPositions);

#if DEBUG
          EncodedTrainingPosition[] uncompressedPositionsTest = new EncodedTrainingPosition[numPositions];
          EncodedTrainingPositionCompressedConverter.Decompress(compressedPositions, uncompressedPositionsTest, numPositions);
          for (int i = 0; i < numPositions; i++)
          {
            if (!encodedTrainingPositions[i].Equals(uncompressedPositionsTest[i]))
            {
              for (int p=0;p<1858;p++)
                if (encodedTrainingPositions[i].Policies[p] != uncompressedPositionsTest[i].Policies[p])
                  Console.WriteLine("FAIL " + i + " " +p + " " + encodedTrainingPositions[i].Policies[p] + " " + uncompressedPositionsTest[i].Policies[p]);
              throw new Exception("Decompress failed, possibly move overflow in EncodedPolicyVectorCompressed.MAX_MOVES");
            }
          }
#endif
          StreamUtils.ZStandardCompressStructArrayToStream(compressedPositions, numPositions, compressedFile, compressionLevel);
          posBytesWritten += Marshal.SizeOf<EncodedTrainingPositionCompressed>() * numPositions;
        }
        else
        {
          // Non compressed file, e.g. LICENSE file in text format.
          compressedFile.Write(decompressedData.ToArray());
        }

        compressedFile.Close();
        tarWriter.WriteEntry(tempFN, entryName.Replace(".gz", ".zst"));
      }

      return tempFN;
    }


    [ThreadStatic]
    static byte[] bufferData;

    private static int ExtractEncodedTrainingPositions(MemoryStream decompressedData,
                                                       Predicate<Memory<EncodedTrainingPosition>> acceptGamePredicate,
                                                       Memory<EncodedTrainingPosition> targetPositions,
                                                       bool writeSentinelAtFirstMoveOfGames)
    {
      if (bufferData == null)
      {
        bufferData = new byte[EncodedTrainingPositionReaderTAREngine.MAX_POSITIONS_PER_STREAM * Marshal.SizeOf<EncodedTrainingPosition>()];
      }

      // Determine number of positions.
      long numBytes = decompressedData.Length;
      int numPositions = (int)(numBytes / Marshal.SizeOf<EncodedTrainingPosition>());
      if (numBytes % Marshal.SizeOf<EncodedTrainingPosition>() != 0)
      {
        throw new Exception("Unexpected size");
      }

      // Make sure target span is large enough.
      if (targetPositions.Length < numPositions)
      {
        throw new Exception("Target span too small");
      }

      // Convert the bytes into array of EncodedTrainingPosition.
      decompressedData.Position = 0;
      decompressedData.Read(bufferData);
      unsafe
      {
        fixed (byte* sourcePtr = &bufferData[0])
        {
          fixed (EncodedTrainingPosition* targetPtr = &targetPositions.Span[0])
          {
            Unsafe.CopyBlock(targetPtr, sourcePtr, (uint)numBytes);
          }
        }
      }


      if (acceptGamePredicate != null && !acceptGamePredicate(targetPositions.Slice(0, numPositions)))
      {
        return 0;
      }

      if (numPositions > 0 && writeSentinelAtFirstMoveOfGames)
      {
        // Mark the first position as the beginning of a game.
        EncodedTrainingPositionCompressedConverter.SetSentinelIsFirstMoveInGame(ref targetPositions.Span[0]);
      }
      return numPositions;
    }

  }
}
