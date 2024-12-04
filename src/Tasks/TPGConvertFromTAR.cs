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

using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.EncodedPositions;

using CeresTrain.UserSettings;
using CeresTrain.TPG.TPGGenerator;
using CeresTrain.TrainingDataGenerator;
using Ceres.Base.Misc;
using Ceres.Chess.UserSettings;

#endregion


namespace CeresTrain.Tasks
{
  /// <summary>
  /// Utility class to convert TAR files to TPG files.
  /// This can be accomplished by either:
  ///   1. Direct conversion from TAR files using method GenerateTPGFromTARs.
  ///   2. Two step conversion:
  ///     - first calling GeneratePackedZSTFromTARs to create packed/recompressed ZST files with same information as TARs
  ///     - then calling GenerateTPG (pointing to the location of the packed/compressed ZST files created above)
  /// </summary>
  public static class TPGConvertFromTAR
  {
    const bool DEBUG = false; // turn this on to dump the diagnostic data for every position (e.g. deblundering/rescoring)
    const bool SMALL = false; // set to True to reduce file sizes by 80% (for testing)

    // WARNING! Modifying the sizing below may possibly result in low-level failures in TPG generation
    //          due to not fully understood assumptions about the number of positions (divisibility requirements).
    //          Switching to "SMALL" mode (see above) is safe, otherwise exercise caution.
    const int OUTPUT_FILE_SIZE_DIVISOR = SMALL ? 5 : 1; // using 5 yields about 200mm positions in a set
    const long NUM_POS_TOTAL = 4096 * 10 * (1640 / OUTPUT_FILE_SIZE_DIVISOR) * 3;
    // END WARNING



    /// <summary>
    /// Generates TPGs from TAR or packed ZST files.
    /// </summary>
    /// <param name="sourceDirectoryTARsOrZSTs"></param>
    /// <param name="targetDirectoryTPGs"></param>
    /// <param name="numSetsToGenerate">number of TPG file sets to generate (each set is about 201mm positions, 16 files and circa 65gb)</param>
    /// <param name="description">description string to write to metadata file</param>
    public static void GenerateTPG(string sourceDirectoryTARsOrZSTs, 
                                   string targetDirectoryTPGs, 
                                   int numSetsToGenerate, 
                                   string description)
    {
      const int POSITION_SKIP_COUNT = 20;

      for (int i = 0; i < numSetsToGenerate; i++)
      {
        GenerateTPG(sourceDirectoryTARsOrZSTs, targetDirectoryTPGs, NUM_POS_TOTAL, DEBUG, description,
                    (EncodedTrainingPositionGame game, int positionIndex, in Position position) => true, // position.PieceCount <= 10,
                    null,
                    positionSkipCount: POSITION_SKIP_COUNT);
      }
    }


    /// <summary>
    /// Performs actual conversion set of of TAR or packed ZST into TARs.
    /// </summary>
    /// <param name="sourceDirectoryTARsOrZSTs"></param>
    /// <param name="targetDirectoryTPGs"></param>
    /// <param name="numPositionsTotal"></param>
    /// <param name="debugMode"></param>
    /// <param name="description"></param>
    /// <param name="acceptRejectDelegate"></param>
    /// <param name="postprocessorDelegate"></param>
    /// <param name="positionSkipCount"></param>
    /// <param name="numRelatedPositionsPerBlock"></param>
    /// <param name="emitPriorMoveWinLoss"></param>
    public static void GenerateTPG(string sourceDirectoryTARsOrZSTs, 
                                   string targetDirectoryTPGs,
                                   long numPositionsTotal, 
                                   bool debugMode,
                                   string description,
                                   TPGGeneratorOptions.AcceptRejectAnnotationDelegate acceptRejectDelegate = null,
                                   TrainingPositionGenerator.PositionPostprocessor postprocessorDelegate = null,
                                   int positionSkipCount = 20,
                                   int numRelatedPositionsPerBlock = 1,
                                   bool emitPriorMoveWinLoss = false)
    {
      ArgumentNullException.ThrowIfNullOrEmpty(sourceDirectoryTARsOrZSTs, nameof(sourceDirectoryTARsOrZSTs));
      ArgumentNullException.ThrowIfNullOrEmpty(targetDirectoryTPGs, nameof(targetDirectoryTPGs));

      if (!Directory.Exists(sourceDirectoryTARsOrZSTs))
      {
        throw new Exception($"Specified TAR/ZST directory {sourceDirectoryTARsOrZSTs} does not exist.");
      }

      if (!Directory.Exists(targetDirectoryTPGs))
      {
        Directory.CreateDirectory(targetDirectoryTPGs);
      }

      bool useTablebases = CeresUserSettingsManager.Settings.TablebaseDirectory != null;
      if (!useTablebases)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, "Warning: DirTablebases not specified in Ceres.json. Endgame tablebases rescoring will not be used in TPG generation.");
      }

      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"GenerateTPG {numPositionsTotal} positions from {sourceDirectoryTARsOrZSTs} into {targetDirectoryTPGs}");
      Console.WriteLine();

      using (new TimingBlock("Generate Training Positions"))
      {
        NNEvaluator nNEvaluator = null; //  NNEvaluator.FromSpecification($"LC0:{NET1}", "GPU:0,1");

        TPGGeneratorOptions options = new TPGGeneratorOptions()
        {
          TargetCompression = System.IO.Compression.CompressionLevel.Optimal, // SmallestSize is maybe 10% smaller, moderately slower

          CeresJSONFileName = CeresTrainUserSettingsManager.Settings.CeresJSONFileName,
          Description = description,
          SourceDirectory = sourceDirectoryTARsOrZSTs,
          FilenameFilter = null, // f => f.Contains("202105"),

          AcceptRejectAnnotater = acceptRejectDelegate,

          AnnotationNNEvaluator = nNEvaluator,

          OutputFormat = TPGGeneratorOptions.OutputRecordFormat.TPGRecord, // note: possibly update FillInHistoryPlanes below
          FillInHistoryPlanes = true,

          EmitPlySinceLastMovePerSquare = false,

          NumRelatedPositionsPerBlock = numRelatedPositionsPerBlock,
          EmitPriorMoveWinLoss = emitPriorMoveWinLoss,

          RescoreWithTablebase = useTablebases,
          NumThreads = debugMode ? 1 : 16 + Math.Min(Environment.ProcessorCount, 50),

          NumConcurrentSets = debugMode ? 1 : 16,
          PositionSkipCount = positionSkipCount,

          AnnotationPostprocessor = postprocessorDelegate,

          PositionMaxFraction = 1.0f / 10_000_000, // extreme deduplication attempted
          NumPositionsTotal = numPositionsTotal,

          Deblunder = TPGGeneratorOptions.DeblunderType.PositionQ,
          DeblunderThreshold = 0.06f,

          Verbose = debugMode,
          TargetFileNameBase = Path.Combine(targetDirectoryTPGs, @$"TPG_{DateTime.Now.Ticks % 100000}"),

          EnablePositionFocus = true, // Currently this simply filters out extreme blunder-impacted postions
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


    /// <summary>
    /// Converts all TAR files (Lc0 training chunks in v6 format) in the source directory 
    /// to more compact ZST files in the target directory (repacked losslessly into zstandard format with TAR over removed).
    /// </summary>
    public static void GeneratePackedZSTFromTARs(string sourceDirectoryTARs, string targetDirectoryPackedZST, string filenameFilter = "*")
    {
      ArgumentNullException.ThrowIfNullOrEmpty(sourceDirectoryTARs, nameof(sourceDirectoryTARs));
      ArgumentNullException.ThrowIfNullOrEmpty(targetDirectoryPackedZST, nameof(targetDirectoryPackedZST)); 

      if (sourceDirectoryTARs.ToUpper() == targetDirectoryPackedZST.ToUpper())
      {
        throw new ArgumentException("Source and target directories must be different");
      }

      if (!Directory.Exists(sourceDirectoryTARs))
      {
        throw new Exception($"Specified TAR {sourceDirectoryTARs} does not exist.");
      }

      if (!Directory.Exists(targetDirectoryPackedZST))
      {
        Directory.CreateDirectory(targetDirectoryPackedZST);
      }

      Console.WriteLine();
      ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"GeneratePackedZSTFromTARs from {sourceDirectoryTARs} into {targetDirectoryPackedZST}");
      Console.WriteLine();

      const bool WRITE_PACKED = true;
      using (new TimingBlock($"Process {sourceDirectoryTARs} into {targetDirectoryPackedZST} "))
      {
        const int MAX_PARALLEL = 32;
        EncodedTrainingPositionRewriter.ConvertTARDir(sourceDirectoryTARs,
                                                      targetDirectoryPackedZST,
                                                      filenameFilter,
                                                      WRITE_PACKED,
                                                      compressionLevel: 10,
                                                      maxParallel: MAX_PARALLEL,
                                                      acceptGamePredicate: (Memory<EncodedTrainingPosition> gamePositions) => true,
                                                      null,
                                                      int.MaxValue);
      }
    }


  }
}
