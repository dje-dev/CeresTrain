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
using Ceres.Chess.EncodedPositions;

#endregion

namespace CeresTrain.TPG.TPGGenerator
{
  /// <summary>
  /// Struct which encapsulates the logic for determining if a given training position
  /// will be accepted into a TPG data file, based on various criteria.
  /// The objectives include:
  ///   - filtering out some of the noisiest positions (where subsequent blunders were extremely large or imbalanced)
  ///   - reducing frequency of positions which are obviously won or drawn in endgame
  ///   - increasing frequency of positions which are "difficult" i.e. value/policy head outputs differed from search results
  ///   
  /// Although shaping the training data distribution as above seems potentially helpful to allow the network to see
  /// more informative positions on average, care is taken to limit the magnitude of the changes.
  /// 
  /// The concern is that modifying the distribution could induce biases in evaluations, and that some of these
  /// seemingly uninformative positions are actually informative.
  /// 
  /// For example, some endgames might have been recognized by the value head and policy heads of 
  /// the fully trained large network used to generate the training data, but are not obvious early in training.
  /// 
  /// To mitigate this bias risk the acceptance probability has a lower bound (PROBABILITY_ACCEPT_BASE) to insure that
  /// some fraction of positions are unconditionally accepted.
  /// </summary>
  public struct TrainingPositionFocusCalculator
  {
    // Two coefficients relating to rejecting excessively noisy training data points.
    public const float THERSHOLD_REJECT_SINGLE_BLUNDER_MAGNITUDE = 0.30f; // Reject position if any single forward blunder exceeds this value
    public const float THERSHOLD_REJECT_BLUNDER_IMBALANCE = 0.60f; // Reject position if forward blunder imbalance exceeds this value

    // Two coefficients relating to favoring positions based on value difference.
    public const float VALUE_DIFF_SLOPE = 1.0f; // slope of probability of accepting position based on value difference
                                                // (e.g. 0.2 head vs. search difference -> 0.6 extra probability likelihood)

    const float PROBABILITY_ACCEPT_BASE = 0.30f; // starting minimum probability of accepting any position

    const int THRESHOLD_EARLY_GAME_PLY = 40;

    public bool ShouldRejectImbalance;
    public bool ShouldRejectSingleBlunder;

    public float ProbabilityContribEarlyMove;
    public float ProbabilityContribFromValueFocus;
    public float ProbabilityContribFromPolicyFocus;
    public float ProbabilityContribIndecisive;
    public float ProbabilityContribFromPriorMoveBlunder;

    public float RandomDraw;


    /// <summary>
    /// Returns a short description of the acceptance-related data and decision for this position.
    /// </summary>
    public string AcceptShortInfoString
    {
      get
      {
        string blunderStr = "   ";
        if (ShouldRejectImbalance)
        {
          return "blun imb                      ";
        }
        else if (ShouldRejectSingleBlunder)
        {
          return "blun one                       ";
        }

        return $" {100 * RandomDraw,2:N0} | " +
          $"{100 * PROBABILITY_ACCEPT_BASE,2:N0}  " +
          $"{100 * ProbabilityContribEarlyMove,2:N0}  " +
          $"{100 * ProbabilityContribIndecisive,2:N0}  " +
          $"{100 * ProbabilityContribFromValueFocus,2:N0}  " +
          $"{100 * ProbabilityContribFromPolicyFocus,2:N0}  " +
          $"{100 * ProbabilityContribFromPriorMoveBlunder,2:N0}  ";
      }
    }


    /// <summary>
    /// Performs the acceptance calculation for the given position.
    /// </summary>
    /// <param name="rescorer"></param>
    /// <param name="indexPlyThisGame"></param>
    /// <returns></returns>
    public bool CalcAcceptPosition(TrainingPositionGeneratorGameRescorer rescorer, int indexPlyThisGame)
    {
      ref readonly EncodedPositionWithHistory trainingPosition = ref rescorer.PositionRef(indexPlyThisGame);
      ref readonly EncodedPositionEvalMiscInfoV6 thisInfoTraining = ref trainingPosition.MiscInfo.InfoTraining;

      float forwardSumPositiveBlunders = rescorer.forwardSumPositiveBlunders[indexPlyThisGame];
      float forwardSumNegativeBlunders = rescorer.forwardSumNegativeBlunders[indexPlyThisGame];
      float forwardMaxSingleNegativeBlunder = rescorer.forwardMaxSingleNegativeBlunder[indexPlyThisGame];
      float forwardMaxSinglePositiveBlunder = rescorer.forwardMaxSinglePositiveBlunder[indexPlyThisGame];

      // We reject those positions for which the game outcome was mostly like impacted
      // by major injected noise in the game generation process.
      // This improves the quality of the Z (game outcome) training target.
      // Impact is measured both in absolute term (magnitude of suboptimality of largest injected blunder)
      // and also in terms of the magnitude of imbalance of positive and negative blunders experienced by each side.
      ShouldRejectImbalance = (MathF.Abs(forwardSumPositiveBlunders - forwardSumNegativeBlunders) > THERSHOLD_REJECT_BLUNDER_IMBALANCE);
      ShouldRejectSingleBlunder = forwardMaxSingleNegativeBlunder > THERSHOLD_REJECT_SINGLE_BLUNDER_MAGNITUDE
                               || forwardMaxSinglePositiveBlunder > THERSHOLD_REJECT_SINGLE_BLUNDER_MAGNITUDE;

      if (ShouldRejectImbalance || ShouldRejectSingleBlunder)
      {
        return false;
      }


#if NOT
      // In a quick test this idea seemed slightly worse.
      bool isObviouslyWinning = thisInfoTraining.BestQ > 0.95 && trainingPosition.BoardsHistory.History_0.RelativePointsUs > 2;
      bool isObviouslyLosing = thisInfoTraining.BestQ < -0.95 && trainingPosition.BoardsHistory.History_0.RelativePointsUs < -2;
      if (isObviouslyWinning || isObviouslyLosing && Random.Shared.NextDouble() < 0.5f)
      {
        return false;
      }
#endif

      return true;

      // NOTE: The more elaborate focus logic attempt below is commented out.
      // Although not known to have bugs, preliminary tests tests were not encouraging.

#if NOT
    const float EARLY_GAME_BONUS = 0.10f;

    const float THRESHOLD_BONUS_PRIOR_MOVE_WAS_BLUNDER = 0.20f;
    const float BONUS_PRIOR_MOVE_WAS_BLUNDER = 0.30f;

    const float INDECISIVE_BONUS = 0.35f;
    const float VALUE_FOCUS_BONUS_MAX = 0.25f;
    const float POLICY_FOCUS_MAX = 0.25f;
    const float POLICY_FOCUS_SLOPE = 0.4f;

      if (indexPlyThisGame > 0)
      {
        ref readonly EncodedPositionWithHistory thisPos = ref rescorer.PositionRef(indexPlyThisGame);
        ref readonly EncodedPositionWithHistory priorPos = ref rescorer.PositionRef(indexPlyThisGame - 1);
        float qChangeVsLastPosition = thisInfoTraining.BestQ - -priorPos.MiscInfo.InfoTraining.BestQ;
        if (qChangeVsLastPosition > THRESHOLD_BONUS_PRIOR_MOVE_WAS_BLUNDER)
        {
          // Emphasize the (few) positions where the prior move was a blunder
          // This is to counteract a possible observed failure of trained nets recognize positions as bad 
          // when the prior move was a blunder.
          ProbabilityContribFromPriorMoveBlunder = BONUS_PRIOR_MOVE_WAS_BLUNDER;
        }
      }

      bool acceptProbabalistically = true;
      if (!float.IsNaN(thisInfoTraining.OriginalQ))
      {
        if (indexPlyThisGame >= 10 && // don't give bonus in earliest moves since most will be common openings
            indexPlyThisGame < THRESHOLD_EARLY_GAME_PLY)
        {
          // The early game is important, but many positions are hard rejected
          // because of high probability of blunders (very large or imbalanced) somehwere subseuqently.
          // To try to avoid excessive rejection, give early moves a bonus.
          ProbabilityContribEarlyMove = EARLY_GAME_BONUS;
        }
        // Bonus only if either search or value head was indecisive.
        const float INDECISIVE_THRESHOLD = 0.90f;
        bool isDecisive = Math.Abs(thisInfoTraining.OriginalQ) > INDECISIVE_THRESHOLD &&
                          Math.Abs(thisInfoTraining.BestQ) > INDECISIVE_THRESHOLD &&
                          Math.Sign(thisInfoTraining.BestQ) == Math.Sign(thisInfoTraining.OriginalQ);

        const int THRESHOLD_ENDGAME_PIECES = 10;
        bool endgameDraw = Math.Abs(thisInfoTraining.OriginalQ) < 0.01 // drawish
                        && Math.Abs(thisInfoTraining.BestQ) < 0.01f
                        && trainingPosition.FinalPosition.PieceCount <= THRESHOLD_ENDGAME_PIECES;
        if (!isDecisive)
        {
          ProbabilityContribIndecisive = INDECISIVE_BONUS;
        }

        float qDiff = Math.Abs(thisInfoTraining.OriginalQ - thisInfoTraining.BestQ);
        ProbabilityContribFromValueFocus = Math.Min(VALUE_FOCUS_BONUS_MAX, VALUE_DIFF_SLOPE * qDiff);

        float kld = thisInfoTraining.KLDPolicy;
        ProbabilityContribFromPolicyFocus = Math.Min(POLICY_FOCUS_MAX, POLICY_FOCUS_SLOPE * kld);

        float probScore = PROBABILITY_ACCEPT_BASE
                        + ProbabilityContribEarlyMove
                        + ProbabilityContribIndecisive
                        + ProbabilityContribFromPolicyFocus
                        + ProbabilityContribFromValueFocus
                        + ProbabilityContribFromPriorMoveBlunder;
        RandomDraw = (float)Random.Shared.NextDouble();
        acceptProbabalistically = probScore > RandomDraw;
      }

      return acceptProbabalistically;
#endif
    }
  }

}

