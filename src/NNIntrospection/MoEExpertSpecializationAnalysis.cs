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

using Ceres.Base.Math;
using Ceres.Chess;

#endregion

namespace CeresTrain.NNIntrospection
{
  /// <summary>
  /// Analyzes distribution of weight allocated to each expert
  /// across a set of positions, with options to partition the positions
  /// based on the expert which received the highest weight for that position.
  /// </summary>
  public class MoEExpertSpecializationAnalysis
  {
    /// <summary>
    /// Index of layer within network to which this analysis relates.
    /// </summary>
    public readonly int LayerNum;


    public int NumAdded => averageExpertWeights.Count;
    
    List<(Position, float[], float[])> averageExpertWeights = new();

    float[,] sumWeightsPerSquarePerExpertCombine;
    float[,] sumWeightsPerExpertPerSquareDispatch;

    float[,] sumWeightsPerPiecePerExpertCombine;
    float[,] sumWeightsPerExpertPerPieceDispatch;
    float[] sumPieceCounts;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="layerNum"></param>
    public MoEExpertSpecializationAnalysis(int layerNum)
    {
      LayerNum = layerNum;
    }


    /// <summary>
    /// Clears all buffered analysis data.
    /// </summary>
    public void Clear()
    {
      averageExpertWeights.Clear();
      Array.Clear(sumWeightsPerSquarePerExpertCombine);
      Array.Clear(sumWeightsPerExpertPerSquareDispatch);
      Array.Clear(sumWeightsPerPiecePerExpertCombine);
      Array.Clear(sumWeightsPerExpertPerPieceDispatch);
      Array.Clear(sumPieceCounts);
    }


    /// <summary>
    /// Adds a position and the weight allocated to each expert for that position.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="combineMatrix"></param>
    public void AddPosition(Position pos, float[,] dispatchMatrix, float[,] combineMatrix)
    {
      if (sumWeightsPerSquarePerExpertCombine == null)
      {
        sumWeightsPerSquarePerExpertCombine = new float[64, combineMatrix.GetLength(1)];
        sumWeightsPerExpertPerSquareDispatch = new float[combineMatrix.GetLength(1), 64];

        sumWeightsPerPiecePerExpertCombine = new float[13, combineMatrix.GetLength(1)];
        sumWeightsPerExpertPerPieceDispatch = new float[combineMatrix.GetLength(1), 13];
        sumPieceCounts = new float[13];
      }

      float[] avgsByExpert = new float[combineMatrix.GetLength(1)];
      float[] avgsBySquare = new float[64];

      for (int squareIndex = 0; squareIndex < 64; squareIndex++)
      {
        for (int expertIndex = 0; expertIndex < combineMatrix.GetLength(1); expertIndex++)
        {
          float thisWeightCombine = combineMatrix[squareIndex, expertIndex];
          float thisWeightDispatch = dispatchMatrix[expertIndex, squareIndex];
          avgsByExpert[expertIndex] += combineMatrix[squareIndex, expertIndex] / combineMatrix.GetLength(0);
          avgsBySquare[squareIndex] += dispatchMatrix[expertIndex, squareIndex] / 64;
          sumWeightsPerSquarePerExpertCombine[squareIndex, expertIndex] += thisWeightCombine;
          sumWeightsPerExpertPerSquareDispatch[expertIndex, squareIndex] += thisWeightDispatch;

          // Update piece statistics.
          Piece thisPiece = pos[new Square(squareIndex)];

          int pieceIndex = ToPieceIndex(pos.SideToMove, thisPiece);
          sumPieceCounts[pieceIndex] += 1;
          sumWeightsPerPiecePerExpertCombine[pieceIndex, expertIndex] += thisWeightCombine;
          sumWeightsPerExpertPerPieceDispatch[expertIndex, pieceIndex] += thisWeightDispatch;
        }
      }

      averageExpertWeights.Add((pos, avgsByExpert, avgsBySquare));
    }


    /// <summary>
    /// Returns the number of experts.
    /// </summary>
    int NumExperts => averageExpertWeights[0].Item2.GetLength(0);


    /// <summary>
    /// Dumps expert analysis to the Console.
    /// </summary>
    public void Dump()
    {
      Console.WriteLine("Soft MoE Analysis - Layer " + LayerNum);
      float[] averageWeightsPerExpert = new float[NumExperts];
      float[] averageWeightsPerSquare = new float[64];

      int[] maxIndices = new int[averageExpertWeights.Count];
      for (int i = 0; i < averageExpertWeights.Count; i++)
      {
        maxIndices[i] = StatUtils.IndexOfMax(averageExpertWeights[i].Item2);
      }

      for (int squareIndex = 0; squareIndex < 64; squareIndex++)
      {
        for (int i = 0; i < averageExpertWeights.Count; i++)
        {
          averageWeightsPerSquare[squareIndex] += averageExpertWeights[i].Item3[squareIndex] / 64;
        }
      }

      // First dump the positions with the highest weight for each expert.
      // Meanwhile also accumulate statistics for both averageWeightsPerExpert and averageWeightsPerSquare.
      for (int expertIndex = 0; expertIndex < NumExperts; expertIndex++)
      {
        Console.WriteLine($"\r\nEXPERT {expertIndex}");
        for (int i = 0; i < averageExpertWeights.Count; i++)
        {
          averageWeightsPerExpert[expertIndex] += averageExpertWeights[i].Item2[expertIndex] / averageExpertWeights.Count;
          if (maxIndices[i] == expertIndex)
          {
            Position thisPos = averageExpertWeights[i].Item1;
            float weight = averageExpertWeights[i].Item2[expertIndex];
            const float MIN_WEIGHT_SHOW_POSITION = 0.12f;
            if (weight > MIN_WEIGHT_SHOW_POSITION)
            {
              Console.Write($"#pieces={thisPos.PieceCount}  weight {weight} for position {thisPos.FEN}  ");
              if (thisPos.PieceCount <= 12)
              {
                foreach (var piece in thisPos.PiecesEnumeration)
                {
                  Console.Write(piece.Piece.Type + " ");
                }
              }
              Console.WriteLine();
            }
          }
        }
      }

      DumpBySquareByExpert("SQUARE AVERAGE EXPERT WEIGHTS (how much weight does a square receive from each expert's output?)", (s, e) => sumWeightsPerSquarePerExpertCombine[s, e], 0.075f);
      DumpBySquareByExpert("EXPERT AVERAGE SQUARE WEIGHTS (how much weight does an expert receive as input from each square?)", (s, e) => sumWeightsPerExpertPerSquareDispatch[e, s], 0.04f);

      Console.WriteLine("\r\nAVERAGE PER PIECE TYPE DISPATCH WEIGHTS");
      for (int i = 0; i < 13; i++)
      {
        Console.Write($"PIECE {FromPieceIndex(i), 15}");
        for (int j=0; j<NumExperts;j++)
        {
          Console.Write($"{100 * sumWeightsPerExpertPerPieceDispatch[j, i] / sumPieceCounts[i],7:N2} ");
        }
        Console.WriteLine();
      }

      Console.WriteLine("\r\nAVERAGE PER PIECE TYPE COMBINE WEIGHTS");
      for (int i = 0; i < 13; i++)
      {
        Console.Write($"PIECE {FromPieceIndex(i), 15}");
        for (int j = 0; j < NumExperts; j++)
        {
          Console.Write($"{100 * sumWeightsPerPiecePerExpertCombine[i, j] / sumPieceCounts[i],7:N2} ");
        }
        Console.WriteLine();
      }

      Console.WriteLine("\r\nAVERAGE SQUARE WEIGHTS");
      for (int i = 0; i < 64; i++)
      {
        Square thisSquare = new Square(i);
        Console.WriteLine($"SQUARE {thisSquare.FileChar}{thisSquare.RankChar} weight {averageWeightsPerSquare[i],6:N2}");
      }

      Console.WriteLine("\r\nAVERAGE EXPERT WEIGHTS");
      for (int i = 0; i < NumExperts; i++)
      {
        Console.WriteLine($"EXPERT {i} weight {averageWeightsPerExpert[i],6:N2}");
      }

    }

    private void DumpBySquareByExpert(string title, Func<int, int, float> getWeightFunc, float minShowWeight)
    {
      Console.WriteLine("\r\n" + title);
      Console.Write("  ");

      for (int i = 0; i < NumExperts; i++)
      {
        Console.Write($"{i,6:N0} ");
      }
      Console.WriteLine();

      for (int squareNum = 0; squareNum < 64; squareNum++)
      {
        Square sq = new Square(squareNum);
        Console.Write(sq.FileChar.ToString() + sq.RankChar.ToString() + " ");
        for (int expertNum = 0; expertNum < NumExperts; expertNum++)
        {
          float value = getWeightFunc(squareNum, expertNum) / NumAdded;
          if (value < minShowWeight)
          {
            Console.Write($"       ");
          }
          else
          {
            if (value > 2 * minShowWeight)
            {
              Console.ForegroundColor = ConsoleColor.Red;
            }
            Console.Write($"{value,6:N2} ");
            Console.ResetColor();
          }
        }
        Console.WriteLine();
      }
    }

    #region Helpers

    static string FromPieceIndex(int index)
    {
      if (index == 0)
      {
        return "empty";
      }
      else
      {
        PieceType type = index < 7 ? (PieceType)(index) : (PieceType)(index - 6);
        return (index < 7 ? "player_" : "opponent_") + type.ToString();
      }
    }

    static int ToPieceIndex(SideType sideToMove, Piece piece)
    {
      if (piece.Type == PieceType.None)
      {
        return 0;
      }
      else
      {
        return piece.Side == sideToMove ? (int)piece.Type : 7 + ((int)piece.Type - 1);
      }
    }


    #endregion
  }
}
