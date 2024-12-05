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

using Ceres.Base.Math.Random;

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
      (float w, float d, float l, float m, float unc, CompressedPolicyVector policysSearch, CompressedPolicyVector policysNet) targets;
      if (node.N < 100)
      {
        throw new Exception("Insufficient search size to extract reliable policy");
      }

      targets = ExtractTrainingTargetsFromNode(tree, node, true);

      // Create the EncodedPolicyVector (start with -1's, then fill in with the actual populated values).
      EncodedPolicyVector epv = default;
      epv.InitilializeAllNegativeOne();
      unsafe
      {
        float* encodedProbs = epv.ProbabilitiesPtr;
        foreach ((EncodedMove Move, float Probability) entry in targets.policysSearch.ProbabilitySummary(0))
        {
          encodedProbs[entry.Move.IndexNeuralNet] = entry.Probability;
        }
      }

      // TODO: Set best and root values below separately
      int bestMoveIndex = epv.BestMove.IndexNeuralNet;
      float q = targets.w - targets.l;
      float kldPolicyTarget = 0.50f; // TODO: try to fill in some meaningful value
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
        kldPolicy: kldPolicyTarget, unused2: default);

      EncodedTrainingPositionMiscInfo miscInfoAll = new(newPosHistory.MiscInfo.InfoPosition, trainingMiscInfo);
      newPosHistory.SetMiscInfo(miscInfoAll);

      EncodedTrainingPosition ret = new EncodedTrainingPosition(version, inputFormat, newPosHistory, epv);

      return ret;
    }


    /// <summary>
    /// Extract training position from specified NNEvaluatorResult.
    /// </summary>
    /// <param name="evaluatorResult"></param>
    /// <param name="evaluatorResultForUncertainty"></param>
    /// <param name="version"></param>
    /// <param name="inputFormat"></param>
    /// <param name="invarianceInfo"></param>
    /// <param name="searchPosition"></param>
    /// <param name="overrideResultToBeWin"></param>
    /// <param name="verbose"></param>
    /// <returns></returns>
    public static EncodedTrainingPosition ExtractFromNNEvalResult(NNEvaluatorResult evaluatorResult, 
                                                                  NNEvaluatorResult? evaluatorResultForUncertainty,
                                                                  int version, int inputFormat, byte invarianceInfo, 
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


      float uncertaintyV = evaluatorResult.UncertaintyV switch
      {
        float value when !float.IsNaN(value) => value,
        _ when evaluatorResultForUncertainty is not null
            && !float.IsNaN(evaluatorResultForUncertainty.Value.UncertaintyV) => evaluatorResultForUncertainty.Value.UncertaintyV,
        _ when evaluatorResultForUncertainty is not null => MathF.Abs(evaluatorResult.V - evaluatorResultForUncertainty.Value.V),
        _ => MathF.Abs(evaluatorResult.V) > 0.80 ? 0.02f : 0.10f // best guess fill in, less uncerainty if seemingly decisive
      };


      float CalculateKLDPolicy()
      {
        float kldPolicy1 = evaluatorResult.Policy.KLDWith(evaluatorResultForUncertainty.Value.Policy);
        float kldPolicy2 = evaluatorResultForUncertainty.Value.Policy.KLDWith(in evaluatorResult.Policy);

        //        Console.WriteLine("KLDAN " + kldPolicy1 + " " + kldPolicy2 + " " + kldPolicy);
        //        Console.WriteLine("KLDAN " + evaluatorResult.Policy);
        //        Console.WriteLine("KLDAN " + evaluatorResultKLD.Value.Policy);

        // Use average of forward and backward KLDs between the policies for the two evaluators.
        return (kldPolicy1 * 0.5f) + (kldPolicy2 * 0.5f);
      }

      float kldPolicy = evaluatorResult.UncertaintyP switch
      {
        float value when !float.IsNaN(value) => value,
        _ when evaluatorResultForUncertainty is not null
            && !float.IsNaN(evaluatorResultForUncertainty.Value.UncertaintyP) => evaluatorResultForUncertainty.Value.UncertaintyP,
        _ when evaluatorResultForUncertainty is not null => CalculateKLDPolicy(),
        _ => 0.50f // best guess fill in, a typical/median value
      };

      float mlh = evaluatorResult.M switch
      {
        _ when overrideResultToBeWin => PLIES_LEFT_IF_LOST,
        float value when !float.IsNaN(value) => value,
        _ when evaluatorResultForUncertainty is not null 
            && !float.IsNaN(evaluatorResultForUncertainty.Value.M) => evaluatorResultForUncertainty.Value.M,
        _ => (40 + searchPosition.FinalPosition.PieceCount * 4) // best guess fill in, a typical/median value
      };

      // UncertaintyV is defined as abs(BestQ - OriginalQ)
      // Need to come up with an OriginalQ such that this value equals our uncertaintyV
      // To avoid Q going outside [-1, 1] range perturb away from the closes bound
      float deltaOriginalQ = q > 0 ? -uncertaintyV : uncertaintyV;
      float originalQ = q + deltaOriginalQ;

      // The originalD is probably not used/needed downstream.
      // So we just try to keep it the same, if this would yield a valid WDL combination.
      // But if this would make the W or L fall outside the legal range of [-1, 1],
      // then we do make an adjustmet to originalD to avoid this.
      float originalD = d;
      float originalW = 0.5f * (1.0f - originalD + originalQ);
      float originalL = 0.5f * (1.0f - originalD - originalQ);
      if (originalW > 1)
      {
        originalD -= originalW - 1;
      }
      else if (originalL > 1)
      {
        originalD -= originalL - 1;
      }

      // Initially set the result WDL to same as the search WDL.
      Span<float> resultWDL = 
        [
          0.5f * (1.0f - d + q),
          d,
          0.5f * (1.0f - d - q)
        ];

      // But the result WDL should actually be a hard game result where just one of WDL is 1.
      // Choose a sampled (randomized) hard outcome based on the WDL probabilties.
      int sampledWDLIndex = ThompsonSampling.Draw(resultWDL, 3);
      float resultW = sampledWDLIndex == 0 ? 1 : 0;
      float resultD = sampledWDLIndex == 1 ? 1 : 0;
      float resultL = sampledWDLIndex == 2 ? 1 : 0;
      float resultQ = resultW - resultL;

      EncodedPositionEvalMiscInfoV6 trainingMiscInfo = new
        (invarianceInfo: invarianceInfo, depResult: default,
         rootQ: q, bestQ: q,
         rootD: d, bestD: d,
         rootM: mlh, bestM: mlh,
         pliesLeft: mlh,
         resultQ: resultQ, resultD: resultD,
         playedQ: q, playedD: d, playedM: mlh,
         originalQ: originalQ, originalD: originalD, originalM: mlh,
         numVisits: 1,
         playedIndex: (short)bestMoveIndex, bestIndex: (short)bestMoveIndex,
         kldPolicy: kldPolicy, unused2: default);

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

      CompressedPolicyVector.Initialize(ref policy, evalResult.Policy.Side, indicies, probabilities, false);

      return (w, d, l, m, u, policy);
    }


    /// <summary>
    /// Extracts training targets (value, policy, etc.) from specified node in a search tree,
    /// using values derived from the search (e.g. empirical visit counts used for the policy).
    /// </summary>
    /// <param name="tree"></param>
    /// <param name="node"></param>
    /// <returns></returns>
    public static (float w, float d, float l, float m, float unc, CompressedPolicyVector policySearch, CompressedPolicyVector policyNet)
      ExtractTrainingTargetsFromNode(MCTSTree tree, in MCTSNode node, bool includePolicySearch)
    {
      float w = node.WAvg;
      float l = node.LAvg;
      float d = 1 - (w + l);
      float m = float.IsNaN(node.MAvg) ? node.InfoRef.MPosition : node.MAvg;
      float u = MathF.Abs((float)node.Q - node.V);

      if (includePolicySearch && node.NumChildrenExpanded == 0)
      {
        throw new Exception("Node has no children, so cannot extract policy");
      }

      // Extract policy from empirical
      CompressedPolicyVector policySearch = default;
      if (includePolicySearch)
      {
        MCTSNodeStructUtils.ExtractPolicyVectorFromVisitDistribution(node.SideToMove, in node.StructRef, ref policySearch);
      }

      // Extracts the neural network policy (undoing the policy softmax that was applied by the search).
      CompressedPolicyVector policyNeuralNet = default;
      MCTSNodeStructUtils.ExtractPolicyVector(tree.Context.ParamsSelect.PolicySoftmax, in node.StructRef, ref policyNeuralNet);

      return (w, d, l, m, u, policySearch, policyNeuralNet);
    }

  }

}
