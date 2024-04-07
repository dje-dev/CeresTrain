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

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.Positions;

using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.LeafExpansion;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  /// <summary>
  /// Static helper methods that construct EncodedTrainingPosition objects 
  /// from either an MCTSNode or an NNEvaluatorResult. 
  /// </summary>
  public static class EncodedTrainingPositionExtractor
  {
    /// <summary>
    /// Extract training position from specified node in a search tree.
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="node"></param>
    /// <param name="version"></param>
    /// <param name="inputFormat"></param>
    /// <param name="invarianceInfo"></param>
    /// <returns></returns>
    public static EncodedTrainingPosition ExtractFromSearchResult(MCTSTree tree, in MCTSNode node,
                                                                  int version, int inputFormat, byte invarianceInfo)
    {
      // Extract training position (boards)
      EncodedPositionWithHistory newPosHistory = default;
      node.Annotation.CalcRawPosition(node, ref newPosHistory);

      //      newPosHistory.SetFromSequentialPositions(.Positions, false); // this also takes care of the misc info

      // Extract training targets
      (float w, float d, float l, float m, float unc, CompressedPolicyVector policy) targets;
      targets = ExtractTrainingTargetsFromNode(tree, node);

      // Create the EncodedPolicyVector (start with -1's, then fill in with the actual populated values).
      EncodedPolicyVector epv = default;
      epv.InitilializeAllNegativeOne();
      unsafe
      {
        float* encodedProbs = epv.ProbabilitiesPtr;
        foreach ((EncodedMove Move, float Probability) entry in targets.policy.ProbabilitySummary(0))
        {
          encodedProbs[entry.Move.IndexNeuralNet] = entry.Probability;
        }
      }

      // TODO: Set best and root values below separately
      int bestMoveIndex = epv.BestMove.IndexNeuralNet;
      float q = targets.w - targets.l;
      EncodedPositionEvalMiscInfoV6 trainingMiscInfo = new
        (
        invarianceInfo: invarianceInfo, depResult: default,
        rootQ: q, bestQ: q, rootD: targets.d, bestD: targets.d,
        rootM: targets.m, bestM: targets.m, pliesLeft: targets.m,
        resultQ: q, resultD: targets.d,
        playedQ: q, playedD: targets.d, playedM: targets.m,
        originalQ: q, originalD: targets.d, originalM: targets.m,
        numVisits: node.N,
        playedIndex: (short)bestMoveIndex, bestIndex: (short)bestMoveIndex,
        unused1: default, unused2: default);

      EncodedTrainingPositionMiscInfo miscInfoAll = new(newPosHistory.MiscInfo.InfoPosition, trainingMiscInfo);
      newPosHistory.SetMiscInfo(miscInfoAll);

      EncodedTrainingPosition ret = new EncodedTrainingPosition(version, inputFormat, newPosHistory, epv);

      return ret;
    }



    /// <summary>
    /// Extract training position from specified NNEvaluatorResult.
    /// </summary>
    /// <param name="evaluatorResult"></param>
    /// <param name="version"></param>
    /// <param name="inputFormat"></param>
    /// <param name="invarianceInfo"></param>
    /// <param name="searchPosition"></param>
    /// <param name="overrideResultToBeWin"></param>
    /// <param name="verbose"></param>
    /// <returns></returns>
    public static EncodedTrainingPosition ExtractFromNNEvalResult(NNEvaluatorResult evaluatorResult, int version, int inputFormat, byte invarianceInfo, 
                                                                  PositionWithHistory searchPosition, bool overrideResultToBeWin, bool verbose)
    {
      EncodedPositionWithHistory newPosHistory = default;
      newPosHistory.SetFromSequentialPositions(searchPosition.Positions, false); // this also takes care of the misc info

      // Set targets for WDL, etc.
      (float w, float d, float l, float m, float unc, CompressedPolicyVector policy) targets;
      const float POLICY_SOFTMAX_APPLIED = 1.0f; // None, we have directly used the evaluator.
      targets = ExtractTrainingTargetsFromNNEvalResult(evaluatorResult, POLICY_SOFTMAX_APPLIED);

      // Create the EncodedPolicyVector (start with -1's, then fill in with the actual populated values).
      EncodedPolicyVector epv = default;
      epv.InitilializeAllNegativeOne();
      unsafe
      {
        float* encodedProbs = epv.ProbabilitiesPtr; // on stack so struct can't move
        foreach ((EncodedMove Move, float Probability) entry in targets.policy.ProbabilitySummary(0))
        {
          encodedProbs[entry.Move.IndexNeuralNet] = entry.Probability;
        }
      }

      int bestMoveIndex = epv.BestMove.IndexNeuralNet;

      const int PLIES_LEFT_IF_LOST = 20; // No exact value available, assume few ply left if definitively lost.
      float w = overrideResultToBeWin ? 1 : targets.w;
      float d = overrideResultToBeWin ? 0 : targets.d;
      float l = overrideResultToBeWin ? 0 : targets.l;
      float q = w - l;
      float m = overrideResultToBeWin ? PLIES_LEFT_IF_LOST : targets.m;

      EncodedPositionEvalMiscInfoV6 trainingMiscInfo = new
        (
        invarianceInfo: invarianceInfo, depResult: default,
        rootQ: q, bestQ: q, rootD: d, bestD: d,
        rootM: m, bestM: m, pliesLeft: m,
        resultQ: q, resultD: d,
        playedQ: q, playedD: d, playedM: m,
        originalQ: q, originalD: d, originalM: m,
        numVisits: 1,
        playedIndex: (short)bestMoveIndex, bestIndex: (short)bestMoveIndex,
        unused1: default, unused2: default);

      EncodedTrainingPositionMiscInfo miscInfoAll = new(newPosHistory.MiscInfo.InfoPosition, trainingMiscInfo);
      newPosHistory.SetMiscInfo(miscInfoAll);
      
      EncodedTrainingPosition ret = new EncodedTrainingPosition(version, inputFormat, newPosHistory, epv);

      return ret;
    }


    /// <summary>
    /// Extracts training targets (value, policy, etc.) from specified NNEvaluatorResult.
    /// </summary>
    /// <param name="evalResult"></param>
    /// <returns></returns>
    public static (float w, float d, float l, float m, float unc, CompressedPolicyVector policy)
      ExtractTrainingTargetsFromNNEvalResult(NNEvaluatorResult evalResult, float policySoftmaxInUse)
    {
      float w = evalResult.W;
      float l = evalResult.L;
      float d = 1 - (w + l);
      float m = evalResult.M;
      float u = evalResult.UncertaintyV;

      // This version is not what we want, it extracts the neural network policy.
      CompressedPolicyVector policy = default;

      int[] indicies = new int[evalResult.Policy.Count];
      float[] probabilities = new float[evalResult.Policy.Count];

      float accProbabilities = 0.0f;
      for (int i = 0; i < evalResult.Policy.Count; i++)
      {
        (EncodedMove Move, float Probability) thisPolicyInfo = evalResult.Policy.PolicyInfoAtIndex(i);

        indicies[i] = (ushort)thisPolicyInfo.Move.IndexNeuralNet;

        // Extract probability, undoing the softmax which had been applied.
        float prob = MathF.Pow(thisPolicyInfo.Probability, policySoftmaxInUse);
        probabilities[i] = prob;
        accProbabilities += prob;
      }

      // Build spans of indices and probabilities (normalized and then encoded).
      float probScaling = 1.0f / accProbabilities;
      for (int i = 0; i < evalResult.Policy.Count; i++)
      {
        probabilities[i] *= probScaling;
      }

      CompressedPolicyVector.Initialize(ref policy, indicies, probabilities, false);

      return (w, d, l, m, u, policy);
    }


    /// <summary>
    /// Extracts training targets (value, policy, etc.) from specified node in a search tree,
    /// using values derived from the search (e.g. empirical visit counts used for the policy).
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="node"></param>
    /// <returns></returns>
    public static (float w, float d, float l, float m, float unc, CompressedPolicyVector policy)
      ExtractTrainingTargetsFromNode(MCTSTree tree, in MCTSNode node)
    {
      float w = node.WAvg;
      float l = node.LAvg;
      float d = 1 - (w + l);
      float m = float.IsNaN(node.MAvg) ? node.InfoRef.MPosition : node.MAvg;
      float u = MathF.Abs((float)node.Q - node.V);

      // Extract policy from empirical
      CompressedPolicyVector policy = default;
      MCTSNodeStructUtils.ExtractPolicyVectorFromVisitDistribution(in node.StructRef, ref policy);

#if NOT
      // This version is not what we want, it extracts the neural network policy.
      CompressedPolicyVector policyVector = default;
      MCTSNodeStructUtils.ExtractPolicyVector(search.Manager.Context.ParamsSelect.PolicySoftmax, in node.StructRef, ref policyVector);
#endif

      return (w, d, l, m, u, policy);
    }

  }

}
