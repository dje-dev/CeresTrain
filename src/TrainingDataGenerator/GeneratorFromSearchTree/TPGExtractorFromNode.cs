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
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using System.Collections.Generic;
using System.Diagnostics;
using Ceres.Chess.EncodedPositions.Basic;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  /// <summary>
  /// Helper methods which extract training data from a training search tree.
  /// </summary>
  public static class TPGExtractorFromNode
  {
    /// <summary>
    /// Extracts a TPGRecord (suitable as a training target) from specified node in specified search tree.
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="node"></param>
    /// <param name="fractionEmpiricalPolicy"></param>
    /// <param name="emitLastPlySinceSquare"></param>
    /// <returns></returns>
    public static TPGRecord ExtractTPGRecordFromNode(MCTSTree tree, 
                                                     in MCTSNode node, 
                                                     float fractionEmpiricalPolicy, 
                                                     bool emitLastPlySinceSquare)
    {
      EncodedPositionWithHistory encodedPosToConvert = default;
      Span<Position> lastSearchPositions = tree.HistoryPositionsForNode(node, 8, false);
      encodedPosToConvert.SetFromSequentialPositions(lastSearchPositions, false);

      // Set targets for WDL, etc.
      (float w, float d, float l, float m, float unc, CompressedPolicyVector policySearch, CompressedPolicyVector policyNet) targets;
      const int MIN_N_FOR_POLICY = 5;
      bool blendInEmpiricalPolicy = node.N > MIN_N_FOR_POLICY;
      targets = EncodedTrainingPositionExtractor.ExtractTrainingTargetsFromNode(tree, node, blendInEmpiricalPolicy);


      List<int> indices = new();
      List<float> probs = new();
      bool includeEmpiricalPolicy = node.N > MIN_N_FOR_POLICY && fractionEmpiricalPolicy > 0;
      foreach ((EncodedMove Move, float Probability) policyEntryNetwork in targets.policyNet.ProbabilitySummary(0))
      {
        indices.Add(policyEntryNetwork.Move.IndexNeuralNet);

        float probability = policyEntryNetwork.Probability; // default value if not blending in empirical policy
        if (includeEmpiricalPolicy)
        {
          int otherPolicyIndex = targets.policySearch.IndexOfMove(policyEntryNetwork.Move);
          if (otherPolicyIndex != -1)
          {
            float empiricalProb = targets.policySearch.PolicyInfoAtIndex(otherPolicyIndex).Probability;
            probability = (1.0f - fractionEmpiricalPolicy) * policyEntryNetwork.Probability +
                          fractionEmpiricalPolicy          * empiricalProb;
          }
        }
        probs.Add(probability);
      }


      CompressedPolicyVector returnBlendedPolicy = default;
      CompressedPolicyVector.Initialize(ref returnBlendedPolicy, node.SideToMove, indices.ToArray(), probs.ToArray(), alreadySorted: false);

      TPGTrainingTargetNonPolicyInfo targetInfo = default;
      (float w, float d, float l) wdl = (targets.w, targets.d, targets.l);
      targetInfo.ResultDeblunderedWDL = wdl;
      targetInfo.ResultNonDeblunderedWDL = wdl;
      targetInfo.BestWDL = wdl;

      targetInfo.IntermediateWDL = default;// gameAnalyzer.intermediateBestWDL[i];
      targetInfo.MLH = TPGRecordEncoding.MLHEncoded(targets.m);
      targetInfo.DeltaQVersusV = targets.unc;

      // This KLD direction is consistent with Lc0 (training_data.cc) which
      // uses empirical visit distribution as P and network policy as Q.
      targetInfo.KLDPolicy = targets.policySearch.KLDWith(targets.policyNet);

      targetInfo.ForwardMinQDeviation = 0; // TODO is this not knowable, or could we descend in the tree to derive a proxy?
      targetInfo.ForwardMaxQDeviation = 0; // TODO is this not knowable, or could we descend in the tree to derive a proxy?

      targetInfo.DeltaQForwardAbs = float.NaN; // TODO is this not knowable, or could we descend in the tree to derive a proxy?
      targetInfo.Source = TPGTrainingTargetNonPolicyInfo.TargetSourceInfo.Training;

      // Construct TPG record from the above
      TPGRecord tpgRecord = default;
      TPGRecordConverter.ConvertToTPGRecord(in encodedPosToConvert, true, default, targetInfo, returnBlendedPolicy,
                                           CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE,
                                           ref tpgRecord, default,
                                           emitLastPlySinceSquare, 0, 0);

      return tpgRecord;
    }

  }

}
