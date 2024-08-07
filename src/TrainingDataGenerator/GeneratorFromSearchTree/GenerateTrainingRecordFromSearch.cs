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

using Xunit;

using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators;
using Ceres.MCTS.Params;
using Ceres.MCTS.Iteration;
using Ceres.APIExamples;

using CeresTrain.TPG;
using CeresTrain.TPG.TPGGenerator;
using Ceres.Chess.Data.Nets;
using System;

#endregion

namespace CeresTrain.TrainingDataGenerator
{

  /// <summary>
  /// Extracts training position from training data file,
  /// conducts search, and writes out resulting root position as training data record.
  /// </summary>
  public class GenerateTrainingRecordFromSearch
  {
    const int VERSION = 6;
    const int INPUT_FORMAT = 1;
    const int INVARIANCE_INFO = 32;
    const int BATCH_SIZE = 4096;
    const int TARGET_NUM_TPG = BATCH_SIZE * 2;

    public static void TestGenerateTrainingRecordFromSearch()
    {
      EncodedPositionWithHistory testPos = default;
      EncodedPositionMiscInfo testInfo = default;

      // Extract one test position from a TAR file.
      const string TAR_FN = @"d:\tar\t80\training-run1-test80-20221227-0417.tar";
      foreach (EncodedTrainingPositionGame game in EncodedTrainingPositionReader.EnumerateGames(TAR_FN, filterOutFRCGames: true))
      {
        string fenFirstPos = game.PositionAtIndex(0).HistoryPosition(0).FEN;
        Assert.Equal(Position.StartPosition.FEN, fenFirstPos);
        testPos = game.PositionAtIndex(0);
        testInfo = game.PositionMiscInfoAtIndex(0);
        break;
      }


      // TODO: make search easier to do !
      NNEvaluatorDef def = new NNEvaluatorDef(NNEvaluatorType.LC0, RegisteredNets.Aliased["T75"].NetSpecificationString, deviceIndex: 0);
      NNEvaluatorSet evaluators = new NNEvaluatorSet(def, false);
      NNEvaluator evaluator = evaluators.Evaluator1;

      TrainingPositionWriter writer = new TrainingPositionWriter("test.tpg", 1, TPGGeneratorOptions.OutputRecordFormat.TPGRecord, true, System.IO.Compression.CompressionLevel.Optimal, TARGET_NUM_TPG, null, null, null, BATCH_SIZE, false, true, true);

      const int NODES_PER_MOVE = 1_000;
      while (true)
      {
        using (new TimingBlock("Search " + NODES_PER_MOVE))
        {
          MCTSearch search = new MCTSearch();
          search.Search(evaluators, new ParamsSelect(), new ParamsSearch(), null, null,
                        testPos.ToPositionWithHistory(8),
                        SearchLimit.NodesPerMove(NODES_PER_MOVE), false, default, null, null, null, null, false); ;

          EncodedTrainingPosition etp = EncodedTrainingPositionExtractor.ExtractFromSearchResult(
            search.Manager.Context.Tree, search.SearchRootNode, VERSION, INPUT_FORMAT, INVARIANCE_INFO);
          EncodedPositionEvalMiscInfoV6 infoTraining = etp.PositionWithBoards.MiscInfo.InfoTraining;

          TrainingPositionWriterNonPolicyTargetInfo target = new();
          target.ResultDeblunderedWDL = infoTraining.ResultWDL;
          target.ResultNonDeblunderedWDL = infoTraining.ResultWDL;
          target.BestWDL = infoTraining.BestWDL;
          target.IntermediateWDL = default;
          target.MLH = TPGRecordEncoding.MLHEncoded(infoTraining.PliesLeft);

          throw new Exception("Possibly need to remediate and set the KLDPolicy and ForwardMinQDeviation properly just below.");
          target.ForwardMinQDeviation = 0;

          target.DeltaQVersusV = infoTraining.Uncertainty;
          target.DeltaQForwardAbs = default;
          target.Source = TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo.Training;

          const int TPG_SET_INDEX = 0;
          writer.Write(in etp, in target, 0, null, CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE, TPG_SET_INDEX);
          writer.Shutdown();

          search.Manager.Dispose(); // TODO: push Dispose into MCTSearch?
        }
      }
    }
  }
}
