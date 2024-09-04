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
using System.Collections.Concurrent;
using System.IO;
using System.IO.Compression;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.NNEvaluators;

#endregion

namespace CeresTrain.TPG.TPGGenerator
{
  /// <summary>
  /// Defines the parameters used when generating training position files
  /// containing pre-shuffled and preprocessed LC0 V6 training data.
  /// </summary>
  public record TPGGeneratorOptions
  {
    public enum DeblunderType
    {
      /// <summary>
      /// No deblundering done.
      /// </summary>
      None,

      /// <summary>
      /// The Q of the best move at the position is substituted if the
      /// move made was suboptimal beyond the threshold.
      /// </summary>
      PositionQ,
    };


    public enum OutputRecordFormat
    {
      /// <summary>
      /// Standard LC0 training record format.
      /// </summary>
      EncodedTrainingPos,

      /// <summary>
      /// Ceres TPGRecord format.
      /// </summary>
      TPGRecord
    }


    /// <summary>
    /// Method called during processing of each raw game position to be possibly included,
    /// allowing position to possibly be rejected (if returns false) and/or modified
    /// (by changing fields of the position in situ in the gamePositions array).
    /// </summary>
    /// <param name="game"></param>
    /// <param name="positionIndex"></param>
    /// <param name="positionIndex"></param>
    /// <returns></returns>
    public delegate bool AcceptRejectAnnotationDelegate(EncodedTrainingPositionGame game, int positionIndex, in Position position);


    #region Ceres configuration

    /// <summary>
    /// Optional full file name to Ceres.json specifying default settings (such as location of tablabase files).
    /// If left null will use default location (current working directory).
    /// </summary>
    public string CeresJSONFileName { init; get; }

    #endregion


    #region Source data

    /// <summary>
    /// Summary description of these options.
    /// </summary>
    public string Description { init; get; }

    /// <summary>
    /// Machine on which data were generated.
    /// </summary>
    public string MachineName { get; } = Environment.MachineName;


    /// <summary>
    /// Creation date/time of options.
    /// </summary>
    public DateTime StartTime { get; } = DateTime.Now;

    /// <summary>
    /// Directory from which the LC0 TAR files containing game chunks are sourced.
    /// </summary>
    public string SourceDirectory { init; get; }

    /// <summary>
    /// Optional filter which can veto certain TAR files from being included.
    /// </summary>
    public Predicate<string> FilenameFilter { init; get; }

    /// <summary>
    /// Number of plies skipped between selected positions.
    /// Note that consecutive selections are interleaved among the
    /// (possibly multiple) concurrent sets.
    /// 
    /// Therefor (for example) if there NumConcurrentSets=10 and 
    /// PositionSkipCount=20 then a game with 100 ply will have
    /// 5 of its positions across all the files in the set,
    /// none in the same file (only if the number of ply in the
    /// game were >200 then a single file might contain more 
    /// than one sample per game, assuming 10 files are written per set).
    /// 
    /// Smaller numbers will increase generator performance
    /// but decrease position diversity.
    /// </summary>
    public int PositionSkipCount { init; get; } = 20;

    /// <summary>
    /// Optionally a number which limits the fraction of repeated positions
    /// to prevent overtraining on any one position (e.g. those from the openings).
    /// 
    /// Using a very small number (such as 1 in 1 million) will result in about 
    /// 7% of positions being skipped (avoid overfitting and improving training efficiency).
    /// </summary>
    public float PositionMaxFraction = 1.0f / 1_000_000f;

    /// <summary>
    /// Minimum number of ply that a position must have to be accepted.
    /// Allowing early (opening) positions seems unhelpful because:
    ///   - there are few in number, this would be just rote memorization
    ///   - training seemingly can become unstable with a "model collapse"
    ///     back to the start position (possibly due to overrepresentation)
    /// </summary>
    public int MinPositionGamePly = 6;

    /// <summary>
    /// Optional filter which can veto certain positions from being included.
    /// </summary>
    public AcceptRejectAnnotationDelegate AcceptRejectAnnotater { init; get; }

    #endregion

    /// <summary>
    /// Enables tablebase rescoring which consults endgame tablebases
    /// and rewrites result Q based on theoretical optimal outcome.
    /// </summary>
    public bool RescoreWithTablebase { init; get; } = false;

    /// <summary>
    /// If number of ply since last move on each square is emitted.
    /// </summary>
    public bool EmitPlySinceLastMovePerSquare = false;

    /// <summary>
    /// Number of positions in a contiguous block which is written.
    /// Typically just 1, but some modes support writing multiple related positions in consecutive slots.
    /// </summary>
    public int NumRelatedPositionsPerBlock = 1;

    /// <summary>
    /// Type of deblundering in use (if any).
    /// </summary>
    public DeblunderType Deblunder { init; get; } = DeblunderType.PositionQ;


    /// <summary>
    /// If position focus should be enabled, which filters out some positions from being converted.
    /// This feature probably improves training results considerably.
    /// 
    /// Currently the method uses this method:
    ///   - rejects if any single blunder  or collective imbalance between sides is extremely large
    ///     indicating the game result target is extremely noisy and likely to be unhelpful/harmful
    ///  Code also exists (but is not enabled because it is suspected to be not helpful) which:
    ///   - upsamples the "harder" positions, i.e. those where value head and search results were different
    ///     or where the policy head had high uncertainty
    /// </summary>
    public bool EnablePositionFocus { init; get; } = true;


    // Minimum probability for a legal move.
    // N.B. Make sure all legal moves have some nonzero probability, for two reasons:
    //        - this allows us to do legal move masking later because any legal move will be >0, and
    //          (the training code strictly assumes this is true)
    //        - this is realistic, even if search did not try this move the probability should still not be zero
    //        - this "encourages" the net to learn which moves are legal,
    //          since this is likely useful information for the other heads (e.g. value)\
    // Values greater than 0.1% are probably not recommended, since in a training game with 1000 nodes
    // possibly one of the moves truly exceeds this fraction of visits.
    // TODO: consider calibrating minimum probability such that if many moves it is somewhat smaller,
    // so we don't have many unlikely moves consuming too much probability
    // (or pushing total too far away from 100%)
    // NOTE: A value of exactly zero is not allowed, since move masking
    //       looks for nonzero values to identify legal moves.
    public float MinProbabilityForLegalMove = CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE;

    /// <summary>
    /// If Deblunder enabled, sets the minimum difference in Q required
    /// for a move to be considered a blunder 
    /// (in cases where move made was other than best, i.e. deliberately injected noise).
    /// </summary>
    public float DeblunderThreshold { init; get; } = 0.06f;


    /// If Deblunder enabled, sets the minimum difference in Q required
    /// for a move to be considered an unintended blunder 
    /// (based on negative change in evaluations see within next 2 moves).
    public float DeblunderUnintnededThreshold { init; get; } = 999;// disabled, found unhelpful, was 0.15f

    /// <summary>
    /// If the value head estimate (win/loss probabilities)
    /// from the prior position in the game should be emitted.
    /// 
    /// In some situations this data may be unavailable in the training data
    /// (first move, prior move a blunder, or NaN in the training data file).
    /// In this case, Win and Loss are both set to 0.
    /// </summary>
    public bool EmitPriorMoveWinLoss { init; get; } = false;

    /// <summary>
    /// If any missing history planes (e.g. in first moves of game)
    /// should be filled in from last position available in history.
    /// </summary>
    public bool FillInHistoryPlanes { init; get; } = false;


    #region Neural network annotation

    /// <summary>
    /// Optional neural network evaluator which is used to evaluate all positions
    /// which are then passed into AnnotationPostprocessor.
    /// </summary>
    public NNEvaluator AnnotationNNEvaluator { init; get; }

    /// <summary>
    /// Optional method that is called before records are written
    /// which includes NN evaluation (if enabled) and allows arbitrary
    /// final modification of training records.
    /// </summary>
    public TrainingPositionGenerator.PositionPostprocessor AnnotationPostprocessor { init; get; }

    #endregion

    #region Processing parameters

    /// <summary>
    /// If verbose output is emitted showing each processed move.
    /// </summary>
    public bool Verbose { init; get; } = false;

    /// <summary>
    /// Number of threads on which positions are processed.
    /// </summary>
    public int NumThreads { init; get; } = 8 + Math.Min(Environment.ProcessorCount, 64);

    #endregion


    #region Output files

    /// <summary>
    /// Format of output records generated.
    /// </summary>
    public OutputRecordFormat OutputFormat = OutputRecordFormat.EncodedTrainingPos;

    /// <summary>
    /// Number of output TPG files which are generated concurrently during each scan of source TAR data.
    /// Higher values increase efficiency (due to greater parallelism).
    /// The default value of 10 allows one of the files to be used as the test set (10% of data).
    /// </summary>
    public int NumConcurrentSets = 10;

    /// <summary>
    /// Total number of training positions to generate across all members of set.
    /// </summary>
    public long NumPositionsTotal { init; get; }

    /// <summary>
    /// Internal batch sizing used.
    /// </summary>
    public int BatchSize { init; get; } = 4096;

    /// <summary>
    /// Number of training positions to generate per file.
    /// </summary>
    public long NumPositionsPerSet => NumPositionsTotal / NumConcurrentSets;

    /// <summary>
    /// If the Zstandard compression library should be used for compression.
    /// </summary>
    public bool UseZstandard = true;

    /// <summary>
    /// Level of compression to use for output files.
    /// 
    /// For Gzip, using Optimal instead of Fastest increases runtime by 1.5x
    /// but produces files which are approximately 0.75 the size,
    /// with no chainge in training speed.
    /// </summary>
    public CompressionLevel TargetCompression { init; get; } = CompressionLevel.Optimal;

    /// <summary>
    /// Base path/name of files to generate.
    /// If null, no files are written (instead they can be acessed via BufferPostprocessorDelegate). 
    /// </summary>    
    public string TargetFileNameBase { init; get; }

    /// <summary>
    /// Optional delegate to which each batch of records is
    /// sent just before writing to files. 
    /// Shutown is initiated if delegate returns false.
    /// </summary>
    /// <param name="tpgRecords"></param>
    /// <returns></returns>
    public Func<TPGRecord[], bool> BufferPostprocessorDelegate;

    #endregion

    #region Files used

    public float SourceFilesSizeMB { internal set; get; }

    public ConcurrentQueue<string> FilesToProcess { internal set; get; }

    #endregion


    /// <summary>
    /// Validates that chosen values are legal.
    /// </summary>
    public void Validate()
    {
      if (MinProbabilityForLegalMove <= 0)
      {
        throw new Exception("MinProbabilityForLegalMove must be greater than zero.");
      }

      if (PositionMaxFraction <= 0)
      {
        throw new Exception("PositionMaxFraction must be greater than 0 (use 1.0 to disable filtering).");
      }
    }


    /// <summary>
    /// Dumps description of options used to generate raw files to a TextWriter.
    /// </summary>
    /// <param name="outStream"></param>
    public void Dump(TextWriter writer, bool dumpFiles)
    {
      writer.WriteLine($"  MachineName                   : {MachineName}");
      writer.WriteLine($"  Description                   : {Description}");
      writer.WriteLine($"  Ceres JSON Path (override)    : {CeresJSONFileName}");
      writer.WriteLine($"  StartTime                     : {StartTime}");
      writer.WriteLine($"  SourceDirectory               : {SourceDirectory}");
      writer.WriteLine($"  FilenameFilter                : {(FilenameFilter != null ? "*Y*" : "No")}");
      writer.WriteLine($"  NumConcurrentSets             : {NumConcurrentSets}");
      writer.WriteLine($"  PositionSkipCount             : {PositionSkipCount}");
      if (PositionMaxFraction >= 1)
      {
        writer.WriteLine($"  PositionMaxFraction           : (none)");
      }
      else
      {
        writer.WriteLine($"  PositionMaxFraction           : {100 * PositionMaxFraction,8:F5}%");
      }
      writer.WriteLine($"  MinPositionGamePly            : {MinPositionGamePly}");

      writer.WriteLine($"  AcceptRejectAnnotater         : {(AcceptRejectAnnotater != null ? "*Yes*" : "No")}");
      writer.WriteLine();
      writer.WriteLine($"  RescoreWithTablebase          : {RescoreWithTablebase}");
      writer.WriteLine($"  FillInHistoryPlanes           : {FillInHistoryPlanes}");
      writer.WriteLine($"  AnnotationNNEvaluator         : {AnnotationNNEvaluator}");
      writer.WriteLine($"  AnnotationPostprocessor       : {(AnnotationPostprocessor != null ? "*Yes* " : "No")}");
      writer.WriteLine($"  EmitPlySinceLastMovePerSquare : {(EmitPlySinceLastMovePerSquare ? "*Yes* " : "No")}");

      Console.WriteLine();
      writer.WriteLine($"  NumRelatedPositionsPerBlock   : {NumRelatedPositionsPerBlock}");
      writer.WriteLine($"  EmitPriorMoveWinLoss          : {(EmitPriorMoveWinLoss ? "*Yes* " : "No")}");

      Console.WriteLine();
      writer.WriteLine($"  MinProbabilityForLegalMove    : {100 * MinProbabilityForLegalMove}%");
      writer.WriteLine($"  Deblunder                     : {Deblunder}");
      writer.WriteLine($"  Deblunder Threshold           : {DeblunderThreshold}");
      writer.WriteLine($"  Deblunder (unint.) Threshold  : {DeblunderUnintnededThreshold}");
      writer.WriteLine($"  EnablePositionFocus           : {EnablePositionFocus}");
      writer.WriteLine();
      writer.WriteLine($"  OutputFormat                  : {OutputFormat}");
      writer.WriteLine($"  NumThreads                    : {NumThreads}");
      writer.WriteLine($"  NumConcurrentSets             : {NumConcurrentSets}");
      writer.WriteLine($"  NumPositionsTotal             : {NumPositionsTotal,14:N0}");
      writer.WriteLine($"  Bytes per position            : {TPGRecord.TOTAL_BYTES,14:N0}");
      writer.WriteLine($"  Batch Size                    : {BatchSize,14:N0}");

      writer.WriteLine($"  TargetCompression             : {TargetCompression}");
      writer.WriteLine($"  TargetFileNameBase            : {TargetFileNameBase}");

      writer.WriteLine();
      writer.WriteLine($"FILES USED ({SourceFilesSizeMB * 0.001f,6:F3} GB)");
      if (dumpFiles)
      {
        foreach (string fn in FilesToProcess)
        {
          writer.WriteLine("  " + fn);
        }
      }
    }

  }
}