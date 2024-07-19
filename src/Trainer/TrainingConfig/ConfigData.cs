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

using System.Text.Json.Serialization;

using CeresTrain.PositionGenerators;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Describes the source of training data and preprocessing steps to be applied.
  /// </summary>
  public readonly record struct ConfigData
  {
    public enum DataSourceType
    {
      PreprocessedFromTAR,
      DirectFromTPG,
      DirectFromPositionGenerator
    }


    /// <summary>
    /// Constructor for the case of reading positions form source files (LC0 or TPG).
    /// </summary>
    /// <param name="sourceType"></param>
    /// <param name="trainingFilesDirectory"></param>
    public ConfigData(DataSourceType sourceType, string trainingFilesDirectory)
    {
      SourceType = sourceType;
      TrainingFilesDirectory = trainingFilesDirectory;
    }

    /// <summary>
    /// Constructor for the case of using a custom Position generator.
    /// </summary>
    /// <param name="positionGenerator"></param>
    public ConfigData(PositionGenerator positionGenerator)
    {
      SourceType = DataSourceType.DirectFromPositionGenerator;
      PositionGenerator = positionGenerator;
    }


    /// <summary>
    /// Default constructor for deserialization.
    /// </summary>
    [JsonConstructorAttribute]
    public ConfigData()
    {
    } 

    /// <summary>
    /// Type of source data used for training data.
    /// </summary>
    public readonly DataSourceType SourceType { get; init; } = DataSourceType.DirectFromPositionGenerator;

    /// <summary>
    /// The generator function used when SourceType == DirectFromPositionGenerator.
    /// </summary>
    [JsonIgnore]
    public readonly PositionGenerator PositionGenerator { get; init; }

    /// <summary>
    /// Directory containing training data files.
    /// </summary>
    public readonly string TrainingFilesDirectory { get; init; }

    /// <summary>
    /// Number of TPG files that will be skipped when generating training data from TPG files.
    /// This can be set to a nonzero value when restarting training for a checkpoint
    /// to avoid reusing the same training data.
    /// </summary>
    public readonly int NumTPGFilesToSkip { get; init; } = 0;

    /// <summary>
    /// Number of positions to skip between selected training positions 
    /// when generating TPG data from TAR files.
    /// Note that consecutively selected positions are typically spread 
    /// across multiple processor threads and sent to different target files, ehancing shuffling).
    /// </summary>
    public readonly int TARPositionSkipCount { get; init; } = 20;

    /// <summary>
    /// Fraction of the WDL (value) targets that are taken from the Q (search at root).
    /// </summary>
    public readonly float FractionQ { get; init; } = 0.0f;

    /// <summary>
    /// Fraction of the WDL (value) target that is smoothed away from true value toward other values.
    /// </summary>
    public readonly float WDLLabelSmoothing { get; init; } = 0.00f;  

    /// <summary>
    /// If the TPG generator should filter out some fraction of the obvious draws and wins.
    /// </summary>
    public readonly bool FilterObviousDrawsWins = true;
  }
}
