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
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

using static TorchSharp.torch;

using Ceres.Base.DataType;
using CeresTrain.TPG;
using CeresTrain.Networks.Transformer;
using TorchSharp;
using CeresTrain.Utils;
using System.Linq;
using Ceres.Chess.EncodedPositions;
using Ceres.Base.Math;
using Ceres.Chess.NNEvaluators.Ceres.TPG;


#endregion

namespace CeresTrain.TPGDatasets
{
  /// <summary>
  /// Manages conversion of a batch of TPGRecords to a dictionary of tensors,
  /// including preparation of training targets by scaling and other adjustments.
  /// </summary>
  internal class TPGBatchToTensorDictConverter
  {
    const int NUM_SQUARES = 64;

    /// <summary>
    /// Fraction of the value target which is taken from Q (search result) 
    /// rather than game result.
    /// </summary>
    public float FractionQ { get; }

    /// <summary>
    /// Size of batches which are generated.
    /// </summary>
    public int BatchSize { get; }

    /// <summary>
    /// Fraction of label smoothing to be applied to WDL targets.
    /// </summary>
    public float WDLLabelSmoothing { get; }

    /// <summary>
    /// Device on which tensors are created.
    /// </summary>
    public Device Device { get; }

    /// <summary>
    /// Data type of tensors.
    /// </summary>
    public ScalarType DataType { get; }


    byte[] bytesSquares;
    float[] bytesMLHOutput;
    float[] bytesUNCOutput;

    static short[] policyIndicesTemp;
    static Half[] policyValuesTemp;

    float[] bytesValueWDLOutput;
    float[] bytesValue2WDLOutput;
    float[] bytesValueWDLQOutput;

    float[] bytesQDeviationLowerOutputTensors;
    float[] bytesQDeviationUpperOutputTensors;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="fractionQ"></param>
    /// <param name="device"></param>
    /// <param name="dataType"></param>
    /// <param name="batchSize"></param>
    /// <param name="wdlLabelSmoothing"></param>
    public TPGBatchToTensorDictConverter(float fractionQ, Device device, ScalarType dataType, int batchSize, float wdlLabelSmoothing)
    {
      FractionQ = fractionQ;
      Device = device;
      DataType = dataType;
      BatchSize = batchSize;

      bytesSquares = new byte[BatchSize * NUM_SQUARES * TPGRecord.BYTES_PER_SQUARE_RECORD];
      bytesMLHOutput = new float[BatchSize];
      bytesUNCOutput = new float[BatchSize];

      bytesQDeviationLowerOutputTensors = new float[BatchSize];
      bytesQDeviationUpperOutputTensors = new float[BatchSize];

      policyIndicesTemp = new short[BatchSize * TPGRecord.MAX_MOVES];
      policyValuesTemp = new Half[BatchSize * TPGRecord.MAX_MOVES];

      bytesValueWDLOutput = new float[BatchSize * 3];
      bytesValue2WDLOutput = new float[BatchSize * 3];
      bytesValueWDLQOutput = new float[BatchSize * 3];
    }


    /// <summary>
    /// Construct a dictionary from TPGRecord[].
    /// </summary>
    /// <param name="theseRecords"></param>
    /// <returns></returns>
    public unsafe Dictionary<string, Tensor> BuildTensorDictFromTPGRecords(TPGRecord[] theseRecords)
    {
      Parallel.For(0, BatchSize, i =>
      {
        ref readonly TPGRecord thisTPG = ref theseRecords[i];

#if DEBUG
        TPGRecordValidation.Validate(in thisTPG);
#endif
        // Copy the square records
        int offsetBytesSquareAndMove = i * NUM_SQUARES * Marshal.SizeOf<TPGSquareRecord>();
        thisTPG.CopySquares(bytesSquares, offsetBytesSquareAndMove);

        float rawMLH = TPGRecordEncoding.MLHDecoded(thisTPG.MLH);
        bytesMLHOutput[i] = rawMLH / NetTransformer.MLH_DIVISOR;

        bytesUNCOutput[i] = MathF.Abs(thisTPG.DeltaQVersusV);

        bytesQDeviationLowerOutputTensors[i] = (float)thisTPG.QDeviationLower;
        bytesQDeviationUpperOutputTensors[i] = (float)thisTPG.QDeviationUpper;

        // The targets NOT expected to be logits.
        bytesValueWDLOutput[i * 3 + 0] = thisTPG.WDLResultDeblundered[0];
        bytesValueWDLOutput[i * 3 + 1] = thisTPG.WDLResultDeblundered[1];
        bytesValueWDLOutput[i * 3 + 2] = thisTPG.WDLResultDeblundered[2];

        bytesValue2WDLOutput[i * 3 + 0] = thisTPG.WDLResultNonDeblundered[0];
        bytesValue2WDLOutput[i * 3 + 1] = thisTPG.WDLResultNonDeblundered[1];
        bytesValue2WDLOutput[i * 3 + 2] = thisTPG.WDLResultNonDeblundered[2];

        bytesValueWDLQOutput[i * 3 + 0] = thisTPG.WDLQ[0];
        bytesValueWDLQOutput[i * 3 + 1] = thisTPG.WDLQ[1];
        bytesValueWDLQOutput[i * 3 + 2] = thisTPG.WDLQ[2];

        if (FractionQ > 0)
        {
          float w1 = 1.0f - FractionQ;
          float w2 = FractionQ;

          bytesValueWDLOutput[i * 3 + 0] = w1 * thisTPG.WDLResultDeblundered[0] + w2 * thisTPG.WDLQ[0];
          bytesValueWDLOutput[i * 3 + 1] = w1 * thisTPG.WDLResultDeblundered[1] + w2 * thisTPG.WDLQ[1];
          bytesValueWDLOutput[i * 3 + 2] = w1 * thisTPG.WDLResultDeblundered[2] + w2 * thisTPG.WDLQ[2];
        }

        // Copy the policy indices and values into the temporary array for all records in the batch
        int policyBaseIndex = i * TPGRecord.MAX_MOVES;
        ReadOnlySpan<short> pIndices = thisTPG.PolicyIndices;
        ReadOnlySpan<Half> pValues = theseRecords[i].PolicyValues;
        pIndices.CopyTo(policyIndicesTemp.AsSpan().Slice(policyBaseIndex, TPGRecord.MAX_MOVES));
        pValues.CopyTo(policyValuesTemp.AsSpan().Slice(policyBaseIndex, TPGRecord.MAX_MOVES));
      });

      // NOTE: from_array is used everywhere below instead of "tensor" static constructor
      // because faster, only the former has the internal "clone" argument set false

      Tensor mlhOutputTensors = from_array(bytesMLHOutput, DataType, Device);
      Tensor uncOutputTensors = from_array(bytesUNCOutput, DataType, Device);

      Tensor policyOutputTensors = default;
      using (var _ = NewDisposeScope())
      {
        Tensor tensorIndices = from_array(policyIndicesTemp, ScalarType.Int16, Device).to(ScalarType.Int64).reshape(BatchSize, TPGRecord.MAX_MOVES);
#if DEBUG
        for (int i = 0; i < BatchSize * TPGRecord.MAX_MOVES; i++)
        {
          if (policyIndicesTemp[i] < 0 || policyIndicesTemp[i] >= EncodedPolicyVector.POLICY_VECTOR_LENGTH)
          {
            throw new Exception("Invalid policy index found before scatter attempted.");
          }
        }
#endif
        Tensor tensorProbs = from_array(policyValuesTemp, DataType, Device).reshape(BatchSize, TPGRecord.MAX_MOVES);

        // Create target policy probability array, and scatter in values from index/value pairs above.
        policyOutputTensors = zeros([BatchSize, 1858], DataType, Device);
        policyOutputTensors = policyOutputTensors.scatter_(1, tensorIndices, tensorProbs).DetachFromDisposeScope();
      }

      Tensor valueWDLOutputTensors = from_array(bytesValueWDLOutput, DataType, Device).reshape([BatchSize, 3]);
      Tensor value2WDLOutputTensors = from_array(bytesValue2WDLOutput, DataType, Device).reshape([BatchSize, 3]);

      Tensor qDeviationLowerOutputTensors = from_array(bytesQDeviationLowerOutputTensors, DataType, Device);
      Tensor qDeviationUpperOutputTensors = from_array(bytesQDeviationUpperOutputTensors, DataType, Device);

      if (WDLLabelSmoothing > 0)
      {
        throw new NotImplementedException("WDLLabelSmoothing current only implemented in PyTorch backend.");

        // TODO: Implement - see Python code for correct logic (tpg_dataset.py).
        using (IDisposable noGrad = torch.no_grad())
        {
          float smoothing = 0.1f;
          long n_classes = valueWDLOutputTensors.size(1);
          Tensor uniform_distribution = torch.full_like(valueWDLOutputTensors, 1.0f / n_classes);

          Tensor smoothed_targets = (1 - smoothing) * valueWDLOutputTensors + smoothing * uniform_distribution;
          //        valueWDLOutputTensors = smoothed_targets;
        }
      }

      Tensor valueWDLQOutputTensors = from_array(bytesValueWDLQOutput, DataType, Device).reshape([BatchSize, 3]);

      // For efficiency (reduce bandwidth requirement), copy to device as bytes.
      Tensor squaresValues = from_array(bytesSquares, ScalarType.Byte, Device).to(DataType).reshape([BatchSize, NUM_SQUARES * TPGRecord.BYTES_PER_SQUARE_RECORD]);

      // Convert to final type and divide to undo the upscaled representation as byte back to true values.
      squaresValues = squaresValues.div_(ByteScaled.SCALING_FACTOR);

      Dictionary<string, Tensor> dict = new()
      {
        {"squares",       squaresValues },
        {"mlh",           mlhOutputTensors },
        {"unc",           uncOutputTensors },
        {"wdl",           valueWDLOutputTensors },
        {"wdlq",          valueWDLQOutputTensors },
        {"policy",        policyOutputTensors },

        {"wdl2",          value2WDLOutputTensors },
        {"q_deviation_lower", qDeviationLowerOutputTensors },
        {"q_deviation_upper", qDeviationUpperOutputTensors },
    };

      return dict;
    }


  }
}
