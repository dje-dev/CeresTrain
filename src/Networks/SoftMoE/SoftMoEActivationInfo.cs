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

namespace CeresTrain.Networks.SoftMoE
{
  /// <summary>
  /// Records the XPhi matrix used to compute the dispatch and combine weights for a Soft-MoE layer.
  /// NOTE: for memory efficiency it would be possible to keep only a single set of the logits (xPhi) 
  ///       and recompute the Softmax operations as needed to derive the dispatch and combine weights.
  /// </summary>
  public record class SoftMoEActivationInfo
  {
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="layerNum"></param>
    /// <param name="dispatchWeights"></param>
    /// <param name="combineWeights"></param>
    public SoftMoEActivationInfo(int layerNum, float[,,] dispatchWeights, float[,,] combineWeights)
    {
      LayerNum = layerNum;
      DispatchWeights = dispatchWeights;
      CombineWeights = combineWeights;
    }

    /// <summary>
    /// The index of the layer from which this activation info was recorded.
    /// </summary>
    public int LayerNum { get; init; }

    /// <summary>
    /// Recorded dispatch weights.
    /// </summary>
    public float[,,] DispatchWeights { get; init; }

    /// <summary>
    /// Recorded combine weights.
    /// </summary>
    public float[,,] CombineWeights { get; init; }
  }
}
