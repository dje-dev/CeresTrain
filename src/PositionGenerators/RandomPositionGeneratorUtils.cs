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
using System.Diagnostics;
using Ceres.Chess;
using Ceres.Chess.MoveGen;

#endregion

namespace CeresTrain.PositionGenerators
{
  /// <summary>
  /// Static helper methods for creating random positions 
  /// containing a specified set of pieces.
  /// 
  /// Most common use is for simple endgames with few pieces.
  /// Other use cases may work, but may be slow to generate
  /// and/or return positions which are 
  /// uncharacteristic of those appearing in  typical chess games.
  /// </summary>
  public static class RandomPositionGeneratorUtils
  {
    /// <summary>
    /// Generates Position from specified set of pieces (such as "KPPkpp").
    /// </summary>
    /// <param name="piecesList"></param>
    /// <returns></returns>
    public static Position GenEndgame(PieceList piecesList) => GenEndgame(piecesList.Pieces);


    /// <summary>
    /// Generates Position from specified array of pieces.
    /// </summary>
    /// <param name="pieces"></param>
    /// <returns></returns>
    public static Position GenEndgame(params Piece[] pieces)
    {
      // Try repeatedly, since some tries may return default
      // if the position generated turned out to be illegal.
      Position pos;
      do
      {
        pos = DoTryGenEndgame(pieces);
      } while (pos == default);

      return pos;
    }


    #region Internal Helpers

    static bool IsValidPawnSquare(int square) => square >= 8 && square < 56;


    [ThreadStatic]
    static Random rand;

    static Position DoTryGenEndgame(params Piece[] pieces)
    {
      if (rand == null)
      {
        rand = new Random((int)DateTime.Now.Ticks);
      }

      Span<int> squares = stackalloc int[pieces.Length];
      squares.Fill(-1);

      // Try to place each piece on a square.
      Span<PieceOnSquare> piecesPlaced = stackalloc PieceOnSquare[pieces.Length];
      for (int i = 0; i < pieces.Length; i++)
      {
        // Search for unoccupied valid square.
        int pos;
        do
        {
          pos = rand.Next(64);
        } while (squares.IndexOf(pos) != -1
             || pieces[i].Type == PieceType.Pawn && !IsValidPawnSquare(pos));

        squares[i] = pos;
        piecesPlaced[i] = (new Square(pos), pieces[i]);
      }

      return TryFinalizePosition(piecesPlaced);
    }


    [ThreadStatic]
    private static Random random;

    static readonly double decayRate = -Math.Log(1 / 700.0) / 48;


    /// <summary>
    /// Generate a draw from a distribution appropriate for a move 50 counter in game positions,
    /// with values between 0 and 48 and higher density at lower values.
    /// </summary>
    /// <returns></returns>
    static int GenerateRandomMove50Value()
    {
      if (random == null)
      {
        random = new Random();
      }

      const int MAX = 48;
      double randomValue = random.NextDouble();
      double adjustedValue = Math.Log(1 - (1 - Math.Exp(-decayRate * MAX)) * randomValue) / -decayRate; // exponential decay of density
      return (int)adjustedValue;
    }


    [ThreadStatic]
    static long finalizeCounter = 0;

    /// <summary>
    /// Converts a proposed set of pieces and tries to place them on chessboard
    /// and return a valid position.
    /// </summary>
    /// <param name="piecesPlaced"></param>
    /// <returns>a valid Position containing specified pieces, or default if unsuccessful.</returns>
    private static Position TryFinalizePosition(Span<PieceOnSquare> piecesPlaced)
    {
      int move50Count = GenerateRandomMove50Value();
      int repetitionCount = random.Next() < 0.01 ? 1 : 0; // occasionally a single repetition
      const int MOVE_NUM = 1;

      PositionMiscInfo miscInfo = new PositionMiscInfo(false, false, false, false, SideType.White,
                                                       move50Count, repetitionCount, MOVE_NUM, PositionMiscInfo.EnPassantFileIndexEnum.FileNone);

      Position pos = new Position(piecesPlaced, miscInfo);

      // For a subset of positions (1 out of 3), set en passant rights randomly (most of which will not be usable).
      const int GEN_EP_MODULO = 3;
      if (finalizeCounter++ % GEN_EP_MODULO == 0)
      {
        PositionMiscInfo.EnPassantFileIndexEnum candidateEPFile = (PositionMiscInfo.EnPassantFileIndexEnum)Random.Shared.Next(8);
        Piece pieceThisFileRank4 = pos.PieceOnSquare(Square.FromFileAndRank((int)candidateEPFile, 4));
        Piece pieceThisFileRank5 = pos.PieceOnSquare(Square.FromFileAndRank((int)candidateEPFile, 5));
        Piece pieceThisFileRank6 = pos.PieceOnSquare(Square.FromFileAndRank((int)candidateEPFile, 6));
        Debug.Assert(pos.SideToMove == SideType.White);

        // Check if black pawn on rank of double move, and two ranks prior are now empty.
        if (pieceThisFileRank4 == Pieces.BlackPawn
          && pieceThisFileRank5.Type == PieceType.None
          && pieceThisFileRank6.Type == PieceType.None)
        {
          // Rebuild the position with this en passant file set.
          miscInfo = new PositionMiscInfo(false, false, false, false, SideType.White, move50Count, repetitionCount, MOVE_NUM, candidateEPFile);
          pos = new Position(piecesPlaced, miscInfo);
        }
      }

      // Return position unless opponent is in check (illegal position).
      MGPosition mgPos = pos.ToMGPosition;
      bool isValid = !MGMoveGen.IsInCheck(in mgPos, true) && MGMoveGen.AtLeastOneLegalMoveExists(in mgPos);
      return isValid ? pos : default;
    }

    #endregion
  }
}
