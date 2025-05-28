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
using System.Linq;
using System.Text.RegularExpressions;
using Ceres.Chess;

#endregion

namespace CeresTrain.PositionGenerators
{
  /// <summary>
  /// Generates random positions containing specified pieces,
  /// arranged randomly on the chessboard (with legality assured).
  /// </summary>
  public sealed record class PositionGeneratorRandomFromPieces : PositionGenerator
  {
    /// <summary>
    /// Set of underlying generators and their assigned probabilities.
    /// </summary>
    public (PositionGenerator generator, float weight, PieceList)[] ComponentGenerators;


    /// <summary>
    /// Constructor from a pieces string with possibly multiple comma separated piece strings (equal weighted).
    /// Each string segment (e.g. KPK) is expanded to include the reverse side (also Kpk)
    /// unless the segment ends with an exclamation mark (e.g. KPK!).
    /// </summary>
    /// <param name="piecesString"></param>
    public PositionGeneratorRandomFromPieces(string piecesString) : base(piecesString)
    {
      ID = piecesString;

      // Create a PieceList from the string (thereby applying validation logic).
      PieceList piecesList =  new PieceList(piecesString);

      // Rewrite the pieces string in expanded format, adding reverse side (unless requested not to).
      // Also impose a canonical ordering of white pieces first.
      // TODO: possibly move this logic into PieceList.
      List<string> expandedPieces = new();
      foreach (string pieces in piecesString.Split(','))
      {
        if (pieces.EndsWith("!"))
        {
          // Explicitly requests piece set reverse side.
          expandedPieces.Add(UpperCaseCharsFirst(pieces[..^1]));
        }
        else
        {
          expandedPieces.Add(UpperCaseCharsFirst(pieces));
          expandedPieces.Add(UpperCaseCharsFirst(new PieceList(pieces).Reversed.PiecesStr));
        }
      }

      // Remove any duplicates from expandedPieces
      expandedPieces = expandedPieces.Distinct().ToList();

      ComponentGenerators = new (PositionGenerator generator, float weight, PieceList)[expandedPieces.Count];

      float weightEach = 1.0f / expandedPieces.Count;

      for (int i=0; i< expandedPieces.Count;i++)
      {
        string pieceString = UpperCaseCharsFirst(expandedPieces[i]);
        ComponentGenerators[i] = ((new PositionGeneratorRandomFromPiecesSingle(pieceString), weightEach, new PieceList(pieceString)));
      }
    }

    static string UpperCaseCharsFirst(string str) => string.Concat(str.OrderBy(c => char.IsUpper(c)? 0 : 1));


    /// <summary>
    /// Constructor from multiple pieces strings with specified weights.
    /// </summary>
    /// <param name="pieces"></param>
    public PositionGeneratorRandomFromPieces(params (string, float)[] pieces) : base(string.Join(",", pieces))
    {
      ID = string.Join(",", pieces);
      ComponentGenerators = new (PositionGenerator, float, PieceList)[pieces.Length];
      for (int i = 0; i < pieces.Length; i++)
      {
        ComponentGenerators[i] = (new PositionGeneratorRandomFromPiecesSingle(pieces[i].Item1), pieces[i].Item2, new PieceList(pieces[i].Item1));
      }       
    }


    /// <summary>
    /// Constructor from multiple pieces strings with equal weight.
    /// </summary>
    /// <param name="pieces"></param>
    public PositionGeneratorRandomFromPieces(params string[] pieces) : this(string.Join(",", pieces))
    {
    }

    /// Parses a string of (pieces, weight) pairs, normalizes the weights to sum to 1.0,
    /// and returns a PositionGeneratorRandomFromPieces instance.
    /// 
    /// Each pair should be in the format: [PiecesString,Weight], with multiple pairs separated by commas.
    /// Example: "[KQPkqp,0.2],[KRPkrp,0.8]"
    /// 
    /// Negative weights are not allowed. If no valid pairs are found, an exception is thrown.
    public static PositionGeneratorRandomFromPieces CreateFromMultiPiecesStr(string piecesStr)
    {
      if (string.IsNullOrWhiteSpace(piecesStr))
      {
        throw new ArgumentException("Input cannot be null or empty.", nameof(piecesStr));
      }

      List<(string, float)> entries = new List<(string, float)>();

      foreach (Match match in Regex.Matches(piecesStr, @"\[\s*([A-Za-z]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"))
      {
        string pieceString = match.Groups[1].Value;
        string weightStr = match.Groups[2].Value;

        if (!float.TryParse(weightStr, out float weight) || weight < 0f)
        {
          throw new ArgumentException($"Invalid or negative weight: {weightStr}");
        }

        entries.Add((pieceString, weight));
      }

      if (entries.Count == 0)
      {
        throw new ArgumentException("No valid piece entries found.");
      }

      float total = 0f;
      foreach ((string _, float weight) in entries)
      {
        total += weight;
      }

      if (total <= 0f)
      {
        throw new ArgumentException("Sum of weights must be positive.");
      }

      (string, float)[] normalized = entries.Select(entry => (entry.Item1, entry.Item2 / total)).ToArray();

      return new PositionGeneratorRandomFromPieces(normalized);
    }


    /// <summary>
    /// Constructor (from a list of generators with specified weights).
    /// </summary>
    /// <param name="description"></param>
    /// <param name="componentGenerators"></param>
    /// <exception cref="Exception"></exception>
    public PositionGeneratorRandomFromPieces(string description, params (PositionGenerator, float, PieceList)[] componentGenerators) : base(description)
    {
      ID = description;
      ComponentGenerators = componentGenerators;

      VerifySumTo1();
    }


    /// <summary>
    /// Returns another position from the generator.
    /// </summary>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public override Position GeneratePosition()
    {
      float random = Random.Shared.NextSingle();
      float sum = 0;

      for (int i = 0; i < ComponentGenerators.Length; i++)
      {
        (PositionGenerator generator, float weight, PieceList pieces) = ComponentGenerators[i];
        sum += weight;
        if (random < sum || i == ComponentGenerators.Length - 1)
        {
          return generator.GeneratePosition();
        }
      }
      throw new Exception("PositionGeneratorRandomFromPiecesMulti internal error");
    }


    #region Helpers

    /// <summary>
    /// Returns if the specified position could be generated by this generator.
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public override bool PositionMatches(in Position pos)
    {
      foreach ((PositionGenerator, float, PieceList) component in ComponentGenerators)
      {
        if (component.Item3.PositionMatches(in pos))
        {
          return true;
        }
      }   
      
      return false;
    }


    private void VerifySumTo1()
    {
      // Verify sum of weights is very close to 1.0.
      float sum = 0;
      foreach ((PositionGenerator, float, PieceList) component in ComponentGenerators)
      {
        sum += component.Item2;
      }

      if (Math.Abs(sum - 1) > 1E-4)
      {
        throw new Exception("PositionGeneratorRandomFromPiecesMulti: sum should be 1.0 but is " + sum);
      }
    }

    #endregion
  }
}


