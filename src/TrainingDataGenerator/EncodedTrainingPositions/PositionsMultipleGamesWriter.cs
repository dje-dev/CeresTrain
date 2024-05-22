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
using System.Formats.Tar;
using System.Diagnostics;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.EncodedPositions;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  public static partial class EncodedTrainingPositionRewriter
  {
    /// <summary>
    /// 
    /// </summary>
    internal class PositionsMultipleGamesWriter
    {
      internal int NumGamesInBuffer = 0;
      internal int NumPosInBuffer = 0;

      internal readonly int CompressionLevel;
      EncodedTrainingPosition[] positions;

      int numBlocksWritten = 0;
      long numBytesWritten = 0;


      internal PositionsMultipleGamesWriter(int maxGamesPerBlock, int compressionLevel)
      {
        const int MAX_AVG_POSITIONS_PER_GAME = 300;
        positions = new EncodedTrainingPosition[maxGamesPerBlock * MAX_AVG_POSITIONS_PER_GAME];
        CompressionLevel = compressionLevel;
      }


      internal void WriteBlock(TarWriter tarWriter)
      {
        //Console.WriteLine(NumGamesInBuffer + "  GAMEWRITE");

        if (NumGamesInBuffer > 0)
        {
          string entryName = "GB_" + numBlocksWritten + ".zst";
          long posBytesWritten = 0;
          string newFN = WriteToTAR(CompressionLevel, ref posBytesWritten,
                                    tarWriter, entryName, null, true, positions.AsSpan().Slice(0, NumPosInBuffer));

          File.Delete(newFN);

          //// DEBUG
          //          DumpBuffer(positions, NumPosInBuffer);
          ////
          NumGamesInBuffer = 0;
          NumPosInBuffer = 0;

          numBlocksWritten++;
          numBytesWritten += posBytesWritten;
        }
      }

      static void DumpBuffer(EncodedTrainingPosition[] positions, int numPosInBuffer)
      {
        int i = 0;
        int posCount = 0;
        Console.WriteLine();
        while (i < numPosInBuffer)
        {
          int j = i + 1;
          while (j < numPosInBuffer && !EncodedTrainingPositionCompressedConverter.IsMarkedFirstMoveInGame(in positions[j]))
          {
            string fen1 = positions[j].PositionWithBoards.FinalPosition.FEN;
            string fen2 = positions[j].PositionWithBoards.FinalPosition.FEN;

            string TEST_FEN = "r1bk2r1/ppp1n1pp/3p2q1/4p3/4Q3/2NP4/PPP1NPPP/1KR4R b k - 3 1";
            if (fen1.StartsWith(TEST_FEN) || fen2.StartsWith(TEST_FEN))
              Console.WriteLine("found bad " + j);

            j++;
          }
          Console.WriteLine(posCount + " " + (j - i) + " " + positions[i].PositionWithBoards.FinalPosition.FEN);
          posCount++;
          i = j;
        }
      }



      /// <summary>
      /// 
      /// </summary>
      /// <param name="decompressedData"></param>
      /// <param name="extraTrainingPosGenerator">if not null then one additional non-played move is taken and evaluated and added to training positions</param>
      /// <param name="extraTrainingPosWasGenerated"></param>
      internal int AddGamePositionsFromStream(MemoryStream decompressedData,
                                              Predicate<Memory<EncodedTrainingPosition>> acceptGamePredicate,
                                              LC0TrainingPosGeneratorFromSingleNNEval extraTrainingPosGenerator,
                                              out bool extraTrainingPosWasGenerated)
      {
        extraTrainingPosWasGenerated = false;
        int numPosLeftInBuffer = positions.Length - NumPosInBuffer;

        // Extract from stream, without any sort of unmirroring.
        int numPosReadThisGame = ExtractEncodedTrainingPositions(decompressedData, acceptGamePredicate,
                                                                 positions.AsMemory(NumPosInBuffer, numPosLeftInBuffer), true);

        if (numPosReadThisGame == 0)
        {
          return 0;
        }

        Span<EncodedTrainingPosition> positionsThisGame = positions.AsMemory(NumPosInBuffer, numPosReadThisGame).Span;

        NumGamesInBuffer++;
        NumPosInBuffer += numPosReadThisGame;

        extraTrainingPosWasGenerated = false;

        if (extraTrainingPosGenerator != null
         && numPosReadThisGame > 5
         && positionsThisGame[0].FinalPosition.Mirrored == Position.StartPosition) // skip if FRC
        {
          // Generate one additional training position.
          EncodedTrainingPosition newTrainingPosition = default;

          // Check for blunder on last move to see if we should make it on the last move.
          // The data on disk was mirrored, now we need to work with it so unmirror into the usual in-memory representation used by Ceres.
          EncodedTrainingPosition lastPos = positionsThisGame[numPosReadThisGame - 1];
          lastPos.MirrorInPlace();
          bool lastPositionWouldHaveBeenBlunder = LC0TrainingPosGeneratorFromSingleNNEval.TrainingPosWasForcedMovePossiblySeriousBlunder(in lastPos.PositionWithBoards);

          if (lastPositionWouldHaveBeenBlunder)
          {
            newTrainingPosition = extraTrainingPosGenerator.GenerateNextTrainingPosition(in lastPos, lastPos.PositionWithBoards.PlayedMove, true, false);
            extraTrainingPosWasGenerated = true;
          }
          else
          {
            // Randomly choose a parent position to generate the new position from.
            // Don't allow the last position because it was considered above (and might be a terminal position).
            int randomIndex = Random.Shared.Next(numPosReadThisGame - 1);
            EncodedTrainingPosition parentPos = positionsThisGame[randomIndex];
            parentPos.MirrorInPlace();

            // Choose another move at random (different from the played move)
            MGMoveList moves = new MGMoveList();
            MGMoveGen.GenerateMoves(parentPos.FinalPosition.ToMGPosition, moves);

            Debug.Assert(moves.MoveExists(parentPos.PositionWithBoards.PlayedMove));

            if (moves.NumMovesUsed > 1)
            {
              MGMove randomMove;
              do
              {
                randomMove = moves.MovesArray[Random.Shared.Next(moves.NumMovesUsed)];
              } while (randomMove == parentPos.PositionWithBoards.PlayedMove);

              newTrainingPosition = extraTrainingPosGenerator.GenerateNextTrainingPosition(in parentPos, randomMove, false, false);

              const bool VERBOSE = false;
              if (VERBOSE)
              {
                float priorQ = parentPos.PositionWithBoards.MiscInfo.InfoTraining.BestQ;
                float newQ = -newTrainingPosition.PositionWithBoards.MiscInfo.InfoTraining.ResultQ;

                string warnStr = newQ > priorQ + 0.4 ? "?" : " ";
                Console.WriteLine(warnStr + " parentPos from our perspective was " + priorQ + " --> " + newQ);
              }

              extraTrainingPosWasGenerated = true;
            }
          }

          if (extraTrainingPosWasGenerated)
          {
            // Add this final position onto the buffer.
            // Mirror it first to match the expected on-disk representation
            // (just like the other positions in the buffer which were read directly without modification).
            newTrainingPosition.MirrorInPlace();
            positions[NumPosInBuffer++] = newTrainingPosition;
          }
        }

        return numPosReadThisGame;
      }

    }

  }
}
