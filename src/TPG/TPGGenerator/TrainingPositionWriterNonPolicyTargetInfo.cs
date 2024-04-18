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


#endregion

namespace CeresTrain.TPG.TPGGenerator
{
  /// <summary>
  /// Set of target training values to be emitted into TPG file 
  /// (everything except policy).
  /// </summary>
  public record struct TrainingPositionWriterNonPolicyTargetInfo
  {
    public enum TargetSourceInfo
    {
      /// <summary>
      /// Ordinary position encountered during game play.
      /// </summary>
      Training,

      /// <summary>
      /// Position was explicitly marked as blunder (non-optimal move deliberately chosen in game play).
      /// </summary>
      NoiseDeblunder,

      /// <summary>
      /// Position was determined to be a blunder based on subsequent game play.
      /// </summary>
      UnintendedDeblunder,

      /// <summary>
      /// Rescored endgame position from tablebase.
      /// </summary>
      Tablebase,
    };


    public (float w, float d, float l) ResultDeblunderedWDL;
    public (float w, float d, float l) ResultNonDeblunderedWDL;
    public (float w, float d, float l) BestWDL;
    public float MLH;
    public float DeltaQVersusV;
    public float ForwardMinQDeviation;
    public (float, float, float) IntermediateWDL;
    public float DeltaQForwardAbs;
    public TargetSourceInfo Source;

    public float ForwardMaxQDeviation;

    /// <summary>
    /// Esimated Q by which the move played to get this position was 
    /// worse than the best move available at the parent.
    /// </summary>
    public float PlayedMoveQSuboptimality;

    // TODO: clean this up, possibly store somewhere else. Technically these are not targets but rather inputs.
    public float ForwardSumPositiveBlunders;
    public float ForwardSumNegativeBlunders;

    /// <summary>
    /// Value head estimate of win probability (from our perspective) at the prior position.
    /// </summary>
    public float PriorPositionWinP;

    /// <summary>
    /// Value head estimate of draw probability (from our perspective) at the prior position.
    /// </summary>
    public float PriorPositionDrawP;

    /// <summary>
    /// Value head estimate of loss probability (from our perspective) at the prior position.
    /// </summary>
    public float PriorPositionLossP;

    /// <summary>
    /// Neural net index (0...1857) of the move played from prior move in game (or -1 if none).
    /// </summary>
    public short PolicyIndexInParent;
  }


}