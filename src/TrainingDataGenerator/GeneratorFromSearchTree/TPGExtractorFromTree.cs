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
using Ceres.Chess.EncodedPositions;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.LeafExpansion;

using CeresTrain.TPG;
using CeresTrain.TPG.TPGGenerator;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  /// <summary>
  /// Helper methods which extract training data from a training search tree.
  /// </summary>
  public static class TPGExtractorFromTree
  {
    /// <summary>
    /// Extracts a TPGRecord (suitable as a training target) from specified node in specified search tree.
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="node"></param>
    /// <param name="emitLastPlySinceSquare"></param>
    /// <param name="emitMoves"></param>
    /// <returns></returns>
    public static TPGRecord ExtractTPGRecordFromNode(MCTSTree tree, in MCTSNode node, bool emitLastPlySinceSquare)
    {
      EncodedPositionWithHistory encodedPosToConvert = default;
      Span<Position> lastSearchPositions = tree.HistoryPositionsForNode(node, 2, false);
      encodedPosToConvert.SetFromSequentialPositions(lastSearchPositions, false);

      // Set targets for WDL, etc.
      (float w, float d, float l, float m, float unc, CompressedPolicyVector policy) targets;
      targets = EncodedTrainingPositionExtractor.ExtractTrainingTargetsFromNode(tree, node);

      TrainingPositionWriterNonPolicyTargetInfo targetInfo = default;
      (float w, float d, float l) wdl = (targets.w, targets.d, targets.l);
      targetInfo.ResultDeblunderedWDL = wdl;
      targetInfo.ResultNonDeblunderedWDL = wdl;
      targetInfo.BestWDL = wdl;

      targetInfo.IntermediateWDL = default;// gameAnalyzer.intermediateBestWDL[i];
      targetInfo.MLH = TPGRecordEncoding.MLHEncoded(targets.m);
      targetInfo.DeltaQVersusV = targets.unc;

      throw new Exception("Possibly need to remediate and set the ForwardMinQDeviation properly just below.");
      targetInfo.ForwardMinQDeviation = 0;

      targetInfo.DeltaQForwardAbs = float.NaN;// gameAnalyzer.deltaQIntermediateBestWDL[i];
      targetInfo.Source = TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo.Training;

      // Construct TPG record from the above
      TPGRecord tpgRecord = default;
      TPGRecordConverter.ConvertToTPGRecord(in encodedPosToConvert, true, default, targetInfo, targets.policy,
                                           CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE,
                                           ref tpgRecord, default,
                                           emitLastPlySinceSquare, 0, 0);

      return tpgRecord;
    }

  }

}
