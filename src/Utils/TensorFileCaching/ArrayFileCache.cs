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
using System.Runtime.InteropServices;
using System.Collections.Generic;

using Ceres.Base.DataType;

#endregion

namespace CeresTrain.Utils.TensorFileCaching
{
  /// <summary>
  /// Manages reading and writing arrays of type T to file.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public class ArrayFileCache<T> : IDisposable where T : unmanaged
  {
    /// <summary>
    /// Name of underlying file to receive cache contents.
    /// </summary>
    public readonly string FileName;

    /// <summary>
    /// Number of items in each array.
    /// </summary>
    public readonly int LengthOfEachArray;


    T[] buffer;
    int numItemsCurrentlyInBuffer;
    BinaryWriter writer;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="fileName"></param>
    /// <param name="lengthOfEachArray"></param>
    /// <param name="numArraysToBufferBeforeWriteToFile"></param>
    public ArrayFileCache(string fileName, int lengthOfEachArray, int numArraysToBufferBeforeWriteToFile = 8)
    {
      FileName = fileName;
      LengthOfEachArray = lengthOfEachArray;

      // Initialize a buffer sufficient in size to hold contents of temporary arrays.
      buffer = new T[lengthOfEachArray * numArraysToBufferBeforeWriteToFile];
      writer = new BinaryWriter(new FileStream(fileName, FileMode.Create));
    }


    /// <summary>
    /// Adds a specified array to the cache.
    /// </summary>
    /// <param name="array"></param>
    /// <exception cref="ArgumentException"></exception>
    public void AddArray(T[] array)
    {
      if (array.Length != LengthOfEachArray)
      {
        throw new ArgumentException($"Array of wrong length, expect {LengthOfEachArray} but received {array.Length}");
      }

      if (numItemsCurrentlyInBuffer + array.Length > buffer.Length)
      {
        Flush();
      }

      array.CopyTo(buffer, numItemsCurrentlyInBuffer);
      numItemsCurrentlyInBuffer += array.Length;
    }


    /// <summary>
    /// Flushes out any pending arrays to the file.
    /// </summary>
    public void Flush()
    {
      if (numItemsCurrentlyInBuffer > 0)
      {
        byte[] bufferBytes = new byte[numItemsCurrentlyInBuffer * Marshal.SizeOf<T>()];
        var bufferCast = MemoryMarshal.Cast<T, byte>(buffer.AsSpan().Slice(0, numItemsCurrentlyInBuffer));
        bufferCast.CopyTo(bufferBytes);
        //Buffer.BlockCopy(bufferCast, 0, bufferBytes, 0, bufferBytes.Length);
        writer.Write(bufferBytes);
        writer.Flush();
        numItemsCurrentlyInBuffer = 0;
      }
    }

    public void Dispose()
    {
      Flush();
      writer.Close();
    }


    /// <summary>
    /// Returns an enumerable of arrays of type T from the file.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="fileName"></param>
    /// <param name="sizeOfEachArray"></param>
    /// <returns></returns>
    public static IEnumerable<T[]> EnumerateArraysInFile<T>(string fileName, int sizeOfEachArray) where T : unmanaged
    {
      using (FileStream fileStream = new FileStream(fileName, FileMode.Open, FileAccess.Read))
      {
        using (BinaryReader reader = new BinaryReader(fileStream))
        {
          byte[] bytes = new byte[sizeOfEachArray * Marshal.SizeOf<T>()];
          T[] array = new T[sizeOfEachArray];
          while (reader.BaseStream.Position != reader.BaseStream.Length)
          {
            bytes = reader.ReadBytes(sizeOfEachArray * Marshal.SizeOf<T>());
            SerializationUtils.DeSerializeArrayIntoBuffer(bytes, bytes.Length, ref array);

            yield return array;
          }
        }
      }
    }

  }

}
