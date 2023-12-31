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

using Ceres.Chess;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  public record struct GeneratorTPGFromSearchTreesParams
  {
    /// <summary>
    /// Name of directory containing source TAR files.
    /// </summary>
    public string SourceTARsDirectoryName;

    /// <summary>
    /// Number of positions to skip between position selections from in each TAR file.
    /// </summary>
    public int TARPosSkipCount; // 15

    /// <summary>
    /// Directory from which TPGs used to "fill in" additional positions.
    /// </summary>
    public string FillInTPGsDirectoryName;

    /// <summary>
    /// Number of TPGs to extract from TPG fill in directory for every generated position.
    /// </summary>
    public int FillInNumTPGPerGeneratedTPG;


    /// <summary>
    /// Total number of TPGs to generate (not including possible fill in TPGs).
    /// </summary>
    public int TargetNumNonFillInTPGToGenerate; // 250_000

    /// <summary>
    /// Size of search tree in nodes for each search.
    /// </summary>
    public int NodesPerSearch; // 10_000

    /// <summary>
    /// Search limits to be used with Stockfish engine.
    /// </summary>

    public SearchLimit SearchLimitSF;

    /// <summary>
    /// Threshold for determining if node has enough visits to be emitted as a training position.
    /// </summary>
    public int MinNodesWriteAsTPG; //1_000


    /// <summary>
    /// Optional nonzero threshold for uncertainty to determine if node should be accepted for root search position.
    /// </summary>
    public float ThresholdUncertaintyRunSearch; // 0.15

    /// <summary>
    /// Name of output TPG file.
    /// </summary>
    public string OutputTPGName;

    /// <summary>
    /// Compression level to use when writing output TPG file.
    /// </summary>
    public int CompressionLevel; // 8

    /// <summary>
    /// Optional predicate filter to determine if an input file with given name should be accepted.
    /// </summary>
    public Predicate<string> PredicateAcceptFile;
  }
}
