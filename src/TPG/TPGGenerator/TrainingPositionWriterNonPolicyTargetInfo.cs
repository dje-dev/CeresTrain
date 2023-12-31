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


    public (float w, float d, float l) ResultWDL;
    public (float w, float d, float l) BestWDL;
    public float MLH;
    public float DeltaQVersusV;
    public (float, float, float) IntermediateWDL;
    public float DeltaQForwardAbs;
    public TargetSourceInfo Source;
  }


}