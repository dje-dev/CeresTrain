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

using System.IO;
using System;

using Ceres.Base.Benchmarking;
using Ceres.Chess.NNEvaluators;
using CeresTrain.TPG.TPGGenerator;
using CeresTrain.UserSettings;

#endregion


namespace CeresTrain.CeresTrainDefaults
{
  /// <summary>
  /// Static class with default values used for generating TPG files.
  /// </summary>
  public static class CeresTrainGenerateTPGDefault
  {
    const bool DEBUG_MODE = false;

    public const int SKIP_COUNT_IF_ALL_POSITIONS = 20;

    public const int SKIP_COUNT_IF_FILTERED_POSITIONS = 8;

    public static void GenerateTPG(string sourceDir, string targetDir,
                                   long numPositionsTotal, 
                                   string description,
                                   TPGGeneratorOptions.AcceptRejectAnnotationDelegate acceptRejectDelegate,
                                   TrainingPositionGenerator.PositionPostprocessor postprocessorDelegate,
                                   int positionSkipCount)
    {
      using (new TimingBlock("Generate Training Positions"))
      {
        NNEvaluator nNEvaluator = null;

        TPGGeneratorOptions options = new TPGGeneratorOptions()
        {
          TargetCompression = System.IO.Compression.CompressionLevel.Optimal, // SmallestSize is maybe 10% smaller, moderately slower

          // TODO: cleanup hardcoded path here
          CeresJSONFileName = CeresTrainUserSettingsManager.Settings.CeresJSONFileName,
          Description = description,
          SourceDirectory = sourceDir,
          FilenameFilter = null,

          AcceptRejectAnnotater = acceptRejectDelegate,

          AnnotationNNEvaluator = nNEvaluator,

          OutputFormat = TPGGeneratorOptions.OutputRecordFormat.TPGRecord, // note: possibly update FillInHistoryPlanes below
          FillInHistoryPlanes = true,

          EmitPlySinceLastMovePerSquare = false,

          RescoreWithTablebase = true,
          NumThreads = DEBUG_MODE ? 1 : 4 + Math.Min(Environment.ProcessorCount, 80), // See degradation of throughput over time if too high (e.g. if 72 total thread)

          NumConcurrentSets = DEBUG_MODE ? 1 : 16,
          PositionSkipCount = positionSkipCount,

          AnnotationPostprocessor = postprocessorDelegate,

          PositionMaxFraction = 1.0f / 1_000_000,
          NumPositionsTotal = numPositionsTotal,

          Deblunder = TPGGeneratorOptions.DeblunderType.PositionQ,
          DeblunderThreshold = 0.06f,

          Verbose = DEBUG_MODE,
          TargetFileNameBase = Path.Combine(targetDir, @$"TPG_{DateTime.Now.Ticks % 100000}")
        };

        // Create the generator and run
        TrainingPositionGenerator tpg = new(options);
        tpg.RunGeneratorLoop();

        Console.WriteLine();
        Console.WriteLine($"SkipModulus {tpg.NumSkippedDueToModulus} "
                        + $"SkipPositionFilter {tpg.NumSkippedDueToPositionFilter} "
                        + $"SkipBadBestInfo {tpg.NumSkippedDueToBadBestInfo}");

      }

    }

  }
}
