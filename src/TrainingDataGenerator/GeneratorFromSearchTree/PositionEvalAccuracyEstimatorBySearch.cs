﻿#region License notice

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

using Ceres.Base.Math;
using Ceres.Base.OperatingSystem;

using Ceres.Chess;
using Ceres.Chess.Positions;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NetEvaluation.Batch;

using Ceres.MCTS.Params;
using Ceres.MCTS.MTCSNodes;
using Ceres.Features.GameEngines;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators.Ceres.TPG;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  /// <summary>
  /// Manages running searches against specified positions and tracking accuracy statistics across evaluations
  /// for comparison against specified target/optimal values.
  /// </summary>
  public class PositionEvalAccuracyEstimatorBySearch
  {
    public enum ValueTargetType { ResultQ, SearchQ, ValueHeadQ };


    /// <summary>
    /// Definition of neural network used by Ceres engine for searches.
    /// </summary>
    public readonly NNEvaluatorDef EvaluatorDef;

    /// <summary>
    /// Ceres engine used for searches.
    /// </summary>
    public readonly GameEngineCeresInProcess CeresEngine;

    /// <summary>
    /// Ceres engine used for searches.
    /// </summary>
    public readonly GameEngineUCI SFEngine;

    /// <summary>
    /// If the game result should be used for value (instead of BestQ), or neural net head. 
    /// Note that using result Q would be is noisy due to lack of deblundering.
    /// </summary>
    public readonly ValueTargetType ValueTarget;

    /// <summary>
    /// If positions in tablebases are ignored by evaluator.
    /// </summary>
    public readonly bool SkipTablebasePositions;

    /// <summary>
    /// Result of last search performed.
    /// </summary>
    public GameEngineSearchResultCeres LastSearchResult;

    /// <summary>
    /// Result of last Stockfish search performed (if any).
    /// </summary>
    public GameEngineSearchResult LastSFSearchResult;


    ISyzygyEvaluatorEngine syzygyEvaluator;

    int numEvaluatedPositions = 0;
    float sumAbsoluteValueDiffsCeres = 0;
    float sumCrossEntropyErrorCeres = 0;
    int sumTopMoveAgree = 0;
    float sumAbsoluteValueDiffsSF = 0;


    /// <summary>
    /// Number of positions evaluated (since last call go ClearStats).
    /// </summary>
    public int NumEvaluatedPositionsCeres => numEvaluatedPositions;

    /// <summary>
    /// Average cross-entropy error.
    /// </summary>
    public float PolicyAverageCrossEntropyErrorCeres => sumCrossEntropyErrorCeres / numEvaluatedPositions;

    /// <summary>
    /// Fraction of top moves that agree.
    /// </summary>
    public float FractionTopMovesAgree => sumTopMoveAgree / (float)numEvaluatedPositions;

    /// <summary>
    /// Average mean absolute error.
    /// </summary>
    public float ValueMeanAbsoluteErrorCeres => sumAbsoluteValueDiffsCeres / numEvaluatedPositions;

    /// <summary>
    /// Average mean absolute error (vs . Stockfish).
    /// </summary>
    public float ValueMeanAbsoluteErrorSF => sumAbsoluteValueDiffsSF / numEvaluatedPositions;


    //
    /// <summary>
    /// Constructor.
    /// 
    /// NOTE: it is typically desirable to skip tablebase positions because the Ceres engine
    ///       may refuse to perform searches on these (e.g. drawn by material).
    /// TODO: improve Ceres engine to make sure it it always does search
    /// </summary>
    /// <param name="evaluatorDef"></param>
    /// <param name="skipTablebasePositions">if positions which are tablebase draw should be skipped</param>
    public PositionEvalAccuracyEstimatorBySearch(NNEvaluatorDef evaluatorDef, bool skipTablebasePositions,
                                                 ValueTargetType valueTarget, bool includeStockfish = false)
    {
      if (includeStockfish)
      {
        SFEngine = (GameEngineUCI)EngineDefStockfish17().CreateEngine();
        Console.WriteLine("Stockfish Engine Started: " + SFEngine);
      }

      EvaluatorDef = evaluatorDef;
      SkipTablebasePositions = skipTablebasePositions;
      ValueTarget = valueTarget;

      ParamsSearch searchParams = new ParamsSearch();
      searchParams.EnableTablebases = false; // want to always go search to we get policy information
      searchParams.EnableInstamoves = false;
      searchParams.EnableSearchExtension = false;
      CeresEngine = new("Ceres", evaluatorDef, searchParams: searchParams);
      CeresEngine.Warmup();

      if (skipTablebasePositions && CeresUserSettingsManager.Settings.TablebaseDirectory != null)
      {
        syzygyEvaluator = SyzygyEvaluatorPool.GetSessionForPaths(CeresUserSettingsManager.Settings.TablebaseDirectory);
      }
    }


    /// <summary>
    /// Clears the running statisics for the evaluator.
    /// </summary>
    public void ClearStats()
    {
      numEvaluatedPositions = 0;
      sumAbsoluteValueDiffsCeres = 0;
      sumCrossEntropyErrorCeres = 0;
      sumAbsoluteValueDiffsSF = 0;
      sumTopMoveAgree = 0;
    }


    /// <summary>
    /// Runs a search against specified position, computes metrics relative to specified training baseline, 
    /// and returns if the search was performed (suitable for evaluation).
    /// </summary>
    /// <param name="posWithHistory"></param>
    /// <param name="searchLimit"></param>
    /// <param name="trainingPos"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public bool DoSearchEvaluation(PositionWithHistory posWithHistory,
                                   SearchLimit searchLimit, SearchLimit searchLimitSF,
                                   in EncodedTrainingPosition trainingPos)
    {
      CompressedPolicyVector targetPolicy = default;
      float[] targetProbabilities = trainingPos.Policies.ProbabilitiesWithNegativeOneZeroed;
      CompressedPolicyVector.Initialize(ref targetPolicy, posWithHistory.FinalPosition.SideToMove, targetProbabilities, false);
      float[] targetProbabilitiesMirrored = targetPolicy.Mirrored.DecodedNoValidate;

      ref readonly EncodedPositionEvalMiscInfoV6 infoTraining = ref trainingPos.PositionWithBoards.MiscInfo.InfoTraining;

      float trainingQ = GetValueTarget(in infoTraining);

      return DoSearchEvaluation(posWithHistory, searchLimit, searchLimitSF, trainingQ, 
                                targetProbabilitiesMirrored, in infoTraining);
    }


    float GetValueTarget(in EncodedPositionEvalMiscInfoV6 infoTraining)
    {
      switch (ValueTarget)
      {
        case ValueTargetType.ValueHeadQ:
          return float.IsNaN(infoTraining.OriginalQ) ? infoTraining.BestQ : infoTraining.OriginalQ;
        case ValueTargetType.ResultQ:
          return infoTraining.ResultQ;
        case ValueTargetType.SearchQ:
          return infoTraining.BestQ;
        default:
          throw new Exception("Unknown value target type " + ValueTarget);
      }

    }



    /// <summary>
    /// Runs a search against specified position, computes metrics relative to specified training baseline, 
    /// and returns if the position was evaluated (suitable for evaluation).
    /// </summary>
    /// <param name="posWithHistory"></param>
    /// <param name="searchLimit"></param>
    /// <param name="trainingQ"></param>
    /// <param name="targetProbabilitiesMirrored"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public bool DoSearchEvaluation(PositionWithHistory posWithHistory,
                                   SearchLimit searchLimit, 
                                   SearchLimit searchLimitSF,
                                   float trainingQ, 
                                   float[] targetProbabilitiesMirrored,
                                   in EncodedPositionEvalMiscInfoV6 infoTraining)
    {
      //      if (posWithHistory.FinalPosition.CalcTerminalStatus() != GameResult.Unknown) { }

      if (SkipTablebasePositions)
      {
        syzygyEvaluator.ProbeWDL(posWithHistory.FinalPosition, out SyzygyWDLScore tbScore, out SyzygyProbeState tbState);
        bool knownTablebasePosition = tbState != SyzygyProbeState.Fail;
        if (knownTablebasePosition)
        {
          return false;
        }
      }

      if (searchLimit.Type == SearchLimitType.NodesPerMove && searchLimit.Value == 1)
      {
        DoEvaluationBySingleEval(posWithHistory, trainingQ, targetProbabilitiesMirrored, infoTraining);
      }
      else
      {
        DoEvaluationBySearch(posWithHistory, searchLimit, searchLimitSF, trainingQ, targetProbabilitiesMirrored);
      }

      numEvaluatedPositions++;

      return true;
    }


    private void DoEvaluationBySingleEval(PositionWithHistory posWithHistory, float trainingQ, float[] targetProbabilitiesMirrored, EncodedPositionEvalMiscInfoV6 infoTraining)
    {
      NNEvaluatorResult evalResult = CeresEngine.Evaluators.Evaluator1.Evaluate(posWithHistory);
      sumAbsoluteValueDiffsCeres += Math.Abs(trainingQ - evalResult.V);

      float[] policyHeadOutputs = evalResult.Policy.Mirrored.DecodedAndNormalized;
      float thisPolicyErrorSearch = StatUtils.SoftmaxCrossEntropy(targetProbabilitiesMirrored, policyHeadOutputs);

      // Subtract off entropy of target policy to focus on error part.
      float targetEntropy = StatUtils.SoftmaxCrossEntropy(targetProbabilitiesMirrored, targetProbabilitiesMirrored);
      thisPolicyErrorSearch -= targetEntropy;

      sumCrossEntropyErrorCeres += thisPolicyErrorSearch;

      bool topMoveMatches = evalResult.Policy.TopMove(posWithHistory.FinalPosition) == ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(infoTraining.BestMove, posWithHistory.FinalPosition.ToMGPosition);
      sumTopMoveAgree += topMoveMatches ? 1 : 0;

      if (LastSFSearchResult != null)
      {
        sumAbsoluteValueDiffsSF += MathF.Abs(trainingQ - LastSFSearchResult.ScoreQ);
      }
    }

    private void DoEvaluationBySearch(PositionWithHistory posWithHistory, SearchLimit searchLimit, SearchLimit searchLimitSF, float trainingQ, float[] targetProbabilitiesMirrored)
    {
      CeresEngine.ResetGame();
      LastSearchResult = CeresEngine.SearchCeres(posWithHistory, searchLimit);

      if (SFEngine != null)
      {
        SFEngine.ResetGame();
        LastSFSearchResult = SFEngine.Search(posWithHistory, searchLimitSF);
#if NOT
        Console.WriteLine("  T: " + Math.Round(trainingQ, 2) 
                        + " S: " + Math.Round(LastSFSearchResult.ScoreQ, 2)
                        + " C: " + Math.Round(LastSearchResult.ScoreQ, 2)
                        + "   Done SF search " + LastSFSearchResult);
#endif
      }

      // Update statistics
      (float w, float d, float l, float m, float unc, CompressedPolicyVector policySearch, CompressedPolicyVector policyNet) trainingTargets;
      if (LastSearchResult.Search.SearchRootNode.N < 10)
      {
        throw new Exception("Insufficient search size to extract reliable policy ");
      }
      trainingTargets = EncodedTrainingPositionExtractor.ExtractTrainingTargetsFromNode(LastSearchResult.Search.Manager.Context.Tree, LastSearchResult.Search.SearchRootNode, true);
      float thisPolicyErrorSearch = StatUtils.SoftmaxCrossEntropy(targetProbabilitiesMirrored, trainingTargets.policySearch.Mirrored.DecodedAndNormalized);

      // Subtract off entropy of target policy to focus on error part.
      float targetEntropy= StatUtils.SoftmaxCrossEntropy(targetProbabilitiesMirrored, targetProbabilitiesMirrored);
      thisPolicyErrorSearch -= targetEntropy;

      if (float.IsNaN(thisPolicyErrorSearch))
      {
        throw new Exception("Search returned no policy " + posWithHistory.FinalPosition.FEN);
      }

      float qdiffCeres = trainingQ - LastSearchResult.ScoreQ;
      sumAbsoluteValueDiffsCeres += MathF.Abs(qdiffCeres);
      sumCrossEntropyErrorCeres += thisPolicyErrorSearch;

      bool topMoveMatches = trainingTargets.policySearch.TopMove(posWithHistory.FinalPosition) == LastSearchResult.BestMove.BestMove;
      sumTopMoveAgree += topMoveMatches ? 1 : 0;

      if (LastSFSearchResult != null)
      {
        float qDiffSF = trainingQ - LastSFSearchResult.ScoreQ;
        sumAbsoluteValueDiffsSF += MathF.Abs(qDiffSF);
      }
    }


    /// <summary>
    /// Extracts position/training information from specified node in last search tree.
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    public TPGRecord ExtractTPGRecordFromNode(MCTSNode node, float fractionEmpiricalPolicy, bool emitLastPlySinceSquare)
      => TPGExtractorFromNode.ExtractTPGRecordFromNode(LastSearchResult.Search.Manager.Context.Tree, node, fractionEmpiricalPolicy, emitLastPlySinceSquare);


    const int SF_NUM_THREADS = 24;

    static string TB_PATH => CeresUserSettingsManager.Settings.TablebaseDirectory;
    static int SF_HASH_SIZE_MB() => HardwareManager.MemorySize > 256L * 1024 * 1024 * 1024 
                                                                  ? 256 : 64;
    static string SF17_EXE => SoftwareManager.IsLinux ? @"/raid/dev/Stockfish/src/stockfish"
                                                      : @"\\synology\dev\chess\engines\stockfish17-windows-x86-64-avx2.exe";


    public static GameEngineDef EngineDefStockfish17(int numThreads = SF_NUM_THREADS, int hashtableSize = -1) =>
  new GameEngineDefUCI("SF17", new GameEngineUCISpec("SF17", SF17_EXE, numThreads,
                       hashtableSize == -1 ? SF_HASH_SIZE_MB() : hashtableSize, TB_PATH, uciSetOptionCommands: null));

  }
}
