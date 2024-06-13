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

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;

#endregion

namespace CeresTrain.Utils
{
  /// <summary>
  /// 
  /// </summary>
  public class PGNFileEnumerator
  {
    /// <summary>
    /// Name of underlying PGN file.
    /// </summary>
    public readonly string PGNFileName;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="pgnFileName"></param>
    /// <exception cref="Exception"></exception>
    public PGNFileEnumerator(string pgnFileName)
    {
      if (!System.IO.File.Exists(pgnFileName))
      {
        throw new Exception($"Specified PGN file not found: {pgnFileName}");
      }

      PGNFileName = pgnFileName;
    }


    /// <summary>
    /// Enumerates all positions in the PGN file as sequential Position objects.\
    /// optionally filtered by acceptFunc and/or a skip count.
    /// </summary>
    /// <param name="acceptFunc">if specified Position should be returned</param>
    /// <param name="skipCount">optional skip modulus to return only every N positions</param>
    /// <returns></returns>
    public IEnumerable<Position> EnumeratePositions(Predicate<Position> acceptFunc = null, int skipCount = -1)
    {
      Ceres.Chess.Textual.PgnFileTools.PgnStreamReader pgnReader = new();
      foreach (Ceres.Chess.Textual.PgnFileTools.GameInfo game in pgnReader.Read(PGNFileName))
      {
        game.Headers.TryGetValue("FEN", out string startFEN);
        MGPosition startPos = startFEN == null ? MGPosition.FromPosition(Position.StartPosition) : MGPosition.FromFEN(startFEN);
        MGPosition curPos = startPos;
        int plyIndex = 0;

        Position curPosAsPosition = curPos.ToPosition;

        if (acceptFunc == null || acceptFunc(curPosAsPosition))
        {
          yield return curPosAsPosition;
        }

        foreach (Ceres.Chess.Textual.PgnFileTools.Move move in game.Moves)
        {
          if (move.HasError)
          {
            Console.WriteLine("HasError " + move.Annotation);
            continue;
          }

          Move m1 = Move.FromSAN(curPos.ToPosition, move.ToAlgebraicString());
          MGMove mgMove = MGMoveConverter.MGMoveFromPosAndMove(curPos.ToPosition, m1);
          curPos.MakeMove(mgMove);
          curPosAsPosition = curPos.ToPosition;

          if (acceptFunc == null || acceptFunc(curPosAsPosition))
          {
            yield return curPosAsPosition;
          }

          if (skipCount == -1 || plyIndex % skipCount != 0)
          {
            plyIndex++;
          }
        }
      }
    }

    /// <summary>
    /// Enumerates sequence of all positions (with their full history within the game)
    /// optionally filtered by acceptFunc.
    /// </summary>
    /// <param name="acceptFunc"></param>
    /// <returns></returns>
    public IEnumerable<PositionWithHistory> EnumeratePositionWithHistory(Predicate<PositionWithHistory> acceptFunc = null)
    {
      Ceres.Chess.Textual.PgnFileTools.PgnStreamReader pgnReader = new();
      foreach (Ceres.Chess.Textual.PgnFileTools.GameInfo game in pgnReader.Read(PGNFileName))
      {
        game.Headers.TryGetValue("FEN", out string startFEN);
        MGPosition startPos = startFEN == null ? MGPosition.FromPosition(Position.StartPosition) : MGPosition.FromFEN(startFEN);
        MGPosition curPos = startPos;
        PositionWithHistory curPositionAndMoves = new PositionWithHistory(startPos);

        if (acceptFunc == null || acceptFunc(curPositionAndMoves))
        {
          yield return curPositionAndMoves;
        }

        foreach (Ceres.Chess.Textual.PgnFileTools.Move move in game.Moves)
        {
          if (move.HasError)
          {
            Console.WriteLine("HasError " + move.Annotation);
            continue;
          }

          try
          {
            Move m1 = Move.FromSAN(curPos.ToPosition, move.ToAlgebraicString());
            MGMove mgMove = MGMoveConverter.MGMoveFromPosAndMove(curPos.ToPosition, m1);
            curPositionAndMoves.AppendMove(mgMove);
            curPos.MakeMove(mgMove);
          }
          catch (Exception exc)
          {
            Console.WriteLine($"Invalid move found in {PGNFileName} position {curPos.ToPosition.FEN} saw move string {move.ToString()}. Skipping position.");
          }

          if (acceptFunc == null || acceptFunc(curPositionAndMoves))
          {
            yield return curPositionAndMoves;
          }
        }
      }
    }

  }
}
