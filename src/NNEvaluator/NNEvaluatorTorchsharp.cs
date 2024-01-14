#region Using directives

using System;
using System.Diagnostics;

using Ceres.Base.DataTypes;
using Ceres.Base.Benchmarking;
using System.Runtime.InteropServices;

using static TorchSharp.torch;

using Ceres.Base.DataType;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.MoveGen;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.MCTS.MTCSNodes.Annotation;

using CeresTrain.TPG;
using CeresTrain.Networks.Transformer;
using CeresTrain.Trainer;
using TorchSharp;
using CeresTrain.TrainCommands;

#endregion

namespace CeresTrain.NNEvaluators
{
  /// <summary>
  /// Subclass of NNEvaluator which uses CeresTrain networks.
  /// </summary>
  public class NNEvaluatorTorchsharp : NNEvaluator
  {
    /// <summary>
    /// Type of engine (direct PyTorch via Torchscript or C# re-implementation).
    /// </summary>
    public readonly NNEvaluatorInferenceEngineType EngineType;
    
    /// <summary>
    /// Device on which executor should execute.
    /// </summary>
    public readonly Device Device;

    /// <summary>
    /// Data type used for inference.
    /// </summary>
    public readonly ScalarType DataType;

    /// <summary>
    /// If the evaluator should be configured to include history in the input planes.
    /// </summary>
    public readonly bool IncludeHistory;

    /// <summary>
    /// If the data indicating last move plies should be used.
    /// </summary>
    public bool LastMovePliesEnabled { init; get; }

    /// <summary>
    /// Underlying evaluator engine.
    /// </summary>
    IModuleNNEvaluator PytorchForwardEvaluator;



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="engineType"></param>
    /// <param name="ceresTransformerNetDef"></param>
    /// <param name="configNetExec"></param>
    /// <param name="lastMovePliesEnabled"></param>
    public NNEvaluatorTorchsharp(NNEvaluatorInferenceEngineType engineType, 
                                 ICeresNeuralNetDef ceresTransformerNetDef,
                                 ConfigNetExecution configNetExec,
                                 bool lastMovePliesEnabled = false)
      : this(engineType, new ModuleNNEvaluatorFromTorchScript(configNetExec with { EngineType = engineType},
             (NetTransformerDef)ceresTransformerNetDef),
             configNetExec.Device, configNetExec.DataType, configNetExec.UseHistory, lastMovePliesEnabled)
    {
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="engineType"></param>
    /// <param name="pytorchForwardEvaluator"></param>
    /// <param name="device"></param>
    /// <param name="dataType"></param>
    /// <param name="includeHistory"></param>
    /// <param name="lastMovePliesEnabled"></param>
    public NNEvaluatorTorchsharp(NNEvaluatorInferenceEngineType engineType, 
                                 IModuleNNEvaluator pytorchForwardEvaluator,
                                 Device device, ScalarType dataType, bool includeHistory,
                                 bool lastMovePliesEnabled = false)
    {
      ArgumentNullException.ThrowIfNull(pytorchForwardEvaluator);

      IncludeHistory = includeHistory;
      EngineType = engineType;

      if (dataType != ScalarType.BFloat16)
      {
        throw new Exception("NNEvaluatorTorchsharp currently only supports BFloat16 data type because it is "
                           + "assumed Ceres nets are trained in this data type and running inference in other  data types "
                           + "types typically results in slight performance degradation (even if other type is higher precision).");
      }

      if (lastMovePliesEnabled && !LastMovePliesTracker.PlyTrackingFeatureEnabled)
      {
        throw new NotImplementedException("Need to rebuild Ceres with LAST_MOVE_PLY_TRACKING defined in LastMovePliesTracker.");
      }

      CeresTrainInitialization.PrepareToUseDevice(device);

      pytorchForwardEvaluator.SetType(dataType);

      // TODO: eliminate this need possibly, instead just pick up from Position
      //       (do conversion to EncodedPosition).
      EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true;

      PytorchForwardEvaluator = pytorchForwardEvaluator;
      LastMovePliesEnabled = lastMovePliesEnabled;
      DataType = dataType;
      Device = device;
    }

    // TODO: make this more restrictive to improve performance/
    public override InputTypes InputsRequired => LastMovePliesEnabled ? InputTypes.AllWithLastMovePlies : InputTypes.All;

    public override bool HasM => true;
    public override bool IsWDL => true;
    public override bool HasUncertaintyV => true;

    public override bool IsEquivalentTo(NNEvaluator evaluator)
    {
      throw new NotImplementedException();
      //      return evaluator is NNEvaluatorTorchsharp && PytorchNetFileName
      //        == (evaluator as NNEvaluatorTorchsharp).PytorchNetFileName;
    }

    const int MAX_BATCH_SIZE = 1024;
    public override int MaxBatchSize => MAX_BATCH_SIZE;

    public override bool PolicyReturnedSameOrderMoveList => false;

    /// <summary>
    /// When true and playing using SearchLimit of BestValueMove, engine using this evaluator 
    /// will slightly adjust evaluation when repetitions are nonzero to prefer repetitions/draws
    /// when seemingly losing and disfavor when seemingly winning.
    /// This feature can compensate for lack of history consideration by the neural network.
    /// </summary>
    public override bool UseBestValueMoveUseRepetitionHeuristic { get; set; } = false;

    public override bool SupportsParallelExecution => false;

    byte[] squareBytesAll = new byte[MAX_BATCH_SIZE * Marshal.SizeOf<TPGSquareRecord>() * 64];
    MGPosition[] mgPos = new MGPosition[MAX_BATCH_SIZE];
    MGMoveList moveList = new MGMoveList();


    public IPositionEvaluationBatch Evaluate(ReadOnlySpan<TPGRecord> tpgRecords)
    {
      if (tpgRecords.Length > MAX_BATCH_SIZE)
      {
        throw new Exception($"NNEvaluatorTorchsharp: requested batch size of {tpgRecords.Length} exceeds maximum supported of {MAX_BATCH_SIZE}");
      }

      for (int i = 0; i < tpgRecords.Length; i++)
      {
        // Extract as bytes.
        tpgRecords[i].CopySquares(squareBytesAll, i * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD);
        mgPos[i] = tpgRecords[i].FinalPosition.ToMGPosition;
      }

      return RunEvalAndExtractResultBatch((i) =>
      {
        moveList.Clear();
        MGMoveGen.GenerateMoves(in mgPos[i], moveList);
        return moveList;
      }, tpgRecords.Length, i => mgPos[i], squareBytesAll);

    }


    static bool haveInitialized = false;
    static bool haveWarned = false;
    public override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (LastMovePliesEnabled && !haveWarned)
      {
        const string ERR = @"Support for LastMovePliesEnabled probably not yet present in DoEvaluateIntoBuffers 
                             need to verify that LastMovePlies in positions is initialized from available history
                             when processing positions from outside a search (e.g. from a UCI command with moves).";
        Console.WriteLine(ERR);
        haveWarned = true;
      }

      if (LastMovePliesEnabled &&  positions is not EncodedPositionBatchFlat)
      {
        throw new NotImplementedException("Internal error, not currently implemented, see notes below");
        // This alternate code below has been developed and tested
        // as a way to properly set the IsDestinationSquareLastMove field.
        // Should be possible to easily implement using this code.
#if NOT
        // Set the IsDestinationSquareLastMove on the appropriate square.
        int baseIndex0 = i * EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES;
        int baseIndex1 = baseIndex0 + EncodedPositionBoard.NUM_PLANES_PER_BOARD;
        EncodedPositionBoard board0 = new(positions.PosPlaneBitmaps.Slice(baseIndex0, EncodedPositionBoard.NUM_PLANES_PER_BOARD));
        board0 = board0.Mirrored;
        EncodedPositionBoard board1 = new(positions.PosPlaneBitmaps.Slice(baseIndex1, EncodedPositionBoard.NUM_PLANES_PER_BOARD));
        board1 = board1.Mirrored;
        var lastMoveInfo = EncodedPositionWithHistory.LastMoveInfoFromSideToMovePerspective(in board0, in board1);

        //const bool SET_LAST_MOVE = true;
        if (SET_LAST_MOVE && lastMoveInfo.pieceType != PieceType.None)
        {
          tpgRecord.Squares[lastMoveInfo.toSquare.SquareIndexStartH1].IsDestinationSquareLastMove = (byte)1;
        }

//        tpgRecord.Dump();
#endif
      }

      MGPosition[] mgPos;
      byte[] squareBytesAll;
      byte[] moveBytesAll;
      short[] legalMoveIndices = null;

      TPGRecordConverter.ConvertPositionsToRawSquareBytes(positions, IncludeHistory, positions.Moves, LastMovePliesEnabled, out mgPos, out squareBytesAll, out legalMoveIndices);

      lastPosition = positions.PositionsBuffer.Span[0];
      IPositionEvaluationBatch batch = RunEvalAndExtractResultBatch(i => positions.Moves.Span[i], positions.NumPos,
                                                                    i => mgPos[i], squareBytesAll, legalMoveIndices);
      if (false) // debug code, test against tablebase if accurate
      {
        if (tbEvaluator == null)
        {
          tbEvaluator = SyzygyEvaluatorPool.GetSessionForPaths(@"e:\sygyzy\5and6man");
        }
        for (int i = 0; i < positions.NumPos; i++)
        {
          if (mgPos[i].ToPosition.PieceCount <= 4
            && new PieceList("KPkp").PositionMatches(mgPos[i].ToPosition))
          {
            //        ISyzygyEvaluatorEngine tbEvaluator = SyzygyEvaluatorPool.GetSessionForPaths(@"e:\sygyzy\5and6man");
            tbEvaluator.ProbeWDL(mgPos[i].ToPosition, out SyzygyWDLScore score, out SyzygyProbeState state);
            int win = CeresTrainCommandTrain.WDLToMostProbableV(batch.GetWinP(i), 1 - (batch.GetWinP(i) + batch.GetLossP(i)), batch.GetLossP(i));
            int winTB = score == SyzygyWDLScore.WDLWin ? 1 : score == SyzygyWDLScore.WDLDraw ? 0 : -1;

            if (win != winTB)
            {
              Console.WriteLine("*** BAD " + mgPos[i].ToPosition.FEN + " TB score: " + score + " " + MathF.Round(batch.GetV(i), 1));
            }
            else
            {
              Console.WriteLine("   ok " + mgPos[i].ToPosition.FEN + " TB score: " + score + " " + MathF.Round(batch.GetV(0), 1) + " " + win);
            }
            Console.WriteLine(lastPosition.FinalPosition.FEN + " TB score: " + score + " " + MathF.Round(batch.GetV(0), 1));
            Console.WriteLine();
          }
        }
      }
      return batch;
    }

    ISyzygyEvaluatorEngine tbEvaluator = null; // For debug code above only

    public static EncodedPositionWithHistory lastPosition;


    /// <summary>
    /// Optional worker method which evaluates batch of positions which are already converted into native format needed by evaluator.
    /// </summary>
    /// <param name="positionsNativeInput"></param>
    /// <param name="usesSecondaryInputs"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public override IPositionEvaluationBatch DoEvaluateNativeIntoBuffers(object positionsNativeInput, bool usesSecondaryInputs,
                                                                         int numPositions, Func<int, int, bool> posMoveIsLegal,
                                                                         bool retrieveSupplementalResults = false)
    {
      Debug.Assert(!retrieveSupplementalResults);
      Debug.Assert(!usesSecondaryInputs);

#if NOT
      ReadOnlySpan<TPGSquareRecord> squareRecords = MemoryMarshal.Cast<byte, TPGSquareRecord>((byte[])positionsNativeInput);
      //ReadOnlyMemory<TPGSquareRecord> squareRecordsMem =    //new ReadOnlyMemory<TPGSquareRecord>(squareRecords);  

      Position PosAtIndex(ReadOnlySpan<TPGSquareRecord> ros, int index)
      {
        ReadOnlySpan<TPGSquareRecord> thisPosSquareRecords = ros.Slice(index * 64, 64);
        const int CURRENT_POS_INDEX = 0;
        Position thisPos = TPGRecord.PositionForSquares(thisPosSquareRecords, CURRENT_POS_INDEX);
        return thisPos;
      }

      CompressedPolicyVector cpv = default;
      const int MAX_MOVES = CompressedPolicyVector.NUM_MOVE_SLOTS;
      PolicyVectorCompressedInitializerFromProbs.InitializeFromProbsArray(cpv, MAX_MOVES, MAX_MOVES, policiesRaw);  

      Func<int, MGMoveList> getMoveListAtIndex = i => PosAtIndex(squareRecords, i).MoveList;
        i => throw new NotImplementedException();
      Func<int, MGPosition> getMGPosAtIndex =
        i => throw new NotImplementedException();
#endif

      // Extract the square records (as byte arrays) from each of the TPGRecords.
      TPGRecord[] recs = (TPGRecord[])positionsNativeInput;
      byte[] rawBytes = new byte[numPositions * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD];
      for (int i = 0; i < numPositions; i++)
      {
        recs[i].CopySquares(rawBytes, i * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD);
      }
      IPositionEvaluationBatch ret = RunEvalAndExtractResultBatch(null, numPositions, null, rawBytes, null);

      return ret;
    }


    static readonly int SQUARE_BYTES_PER_POSITION = 64 * TPGRecord.BYTES_PER_SQUARE_RECORD;

    static byte[] lastBytes = null;


    public IPositionEvaluationBatch RunEvalAndExtractResultBatch(Func<int, MGMoveList> getMoveListAtIndex,
                                                                 int numPositions,
                                                                 Func<int, MGPosition> getMGPosAtIndex,
                                                                 byte[] squareBytesAll,
                                                                 short[] legalMovesIndices = null)
    {
      if (false)
      {
        // Test code to show any differences in raw input compared to last call (only first position checked).
        var subBytes = squareBytesAll.AsSpan().Slice(0, 64 * 135);
        var posx = TPGRecord.PositionForSquares(MemoryMarshal.Cast<byte, TPGSquareRecord>(subBytes), 0, true);
        if (lastBytes != null)
        {
          Console.WriteLine("testcompare vs first seen");
          for (int i = 0; i < 64; i++)
            for (int j = 0; j < 135; j++)
            {
              if (squareBytesAll[i * 135 + j] != lastBytes[i * 135 + j])
              {
                Console.WriteLine("diff at " + i + " " + j + "  ... breaking");
                break;
              }
            }
        }
        if (lastBytes == null)
        {
          lastBytes = new byte[8640];
          Array.Copy(squareBytesAll, lastBytes, lastBytes.Length);
        }
      }

      if (numPositions > MAX_BATCH_SIZE)
      {
        throw new Exception($"NNEvaluatorTorchsharp: requested batch size of {numPositions} exceeds maximum supported of {MAX_BATCH_SIZE}");
      }   

      Tensor predictionValue;
      Tensor predictionPolicy;
      Tensor predictionMLH;
      Tensor predictionUNC;
      FP16[] extraStats0;
      FP16[] extraStats1;

      using (no_grad())
      {
        // Create a Tensor of bytes still on CPU.
        Tensor cpuTensor = tensor(squareBytesAll, [numPositions, 64, TPGRecord.BYTES_PER_SQUARE_RECORD]);

        // Move Tensor to the desired device and data type.
        Tensor inputSquares = cpuTensor.to(Device).to(DataType);

        // Apply scaling factor to TPG square inputs.
        inputSquares = inputSquares.div_(ByteScaled.SCALING_FACTOR);

        // Evaluate using neural net.
        (predictionValue, predictionPolicy, predictionMLH, predictionUNC, extraStats0, extraStats1) = PytorchForwardEvaluator.forwardValuePolicyMLH_UNC(inputSquares, null);//, inputMoves.to(DeviceType, DeviceIndex));

        cpuTensor.Dispose();
        inputSquares.Dispose();
      }

      using (var _ = NewDisposeScope())
      {
        // Subtract the max from logits and exponentiate (in Float32 to preserve accuracy during exponentiation and division).
        Tensor valueFloat = predictionValue.to(ScalarType.Float32);

//        float[] valueOnCPURAW = predictionValue.cpu().FloatArray();
//Console.WriteLine("RAWVAL " + valueOnCPURAW[0] + " " + valueOnCPURAW[1] + " " + valueOnCPURAW[2]);

        Tensor max_logits = torch.max(valueFloat, dim: 1, keepdim: true).values;
        Tensor exp_logits = torch.exp(valueFloat - max_logits);

        // Sum the exponentiated logits along the last dimension and use to normalize.
        Tensor sum_exp_logits = torch.sum(exp_logits, dim: 1, keepdim: true);
        Tensor wdlProbabilities = (exp_logits / sum_exp_logits);
        Span<Half> wdlProbabilitiesCPU = MemoryMarshal.Cast<byte, Half>(wdlProbabilities.to(ScalarType.Float16).cpu().bytes);

        // Cast data to desired C# data type and transfer to CPU.
        // Then get ReadOnlySpans over the underlying data of the tensors.
        // Note: Because this all happens within an DisposeScope, the tensors will be kept alive (not disposed)
        //       for the duration of the block therefore the Spans (which reference underlying Tensor memory at a fixed location)
        //       will be valid for the duration of the block.
        //        ReadOnlySpan<Half> predictionsValue = MemoryMarshal.Cast<byte, Half>(predictionValue.to(ScalarType.Float16).cpu().bytes);
        ReadOnlySpan<Half> predictionsMLH = MemoryMarshal.Cast<byte, Half>(predictionMLH.to(ScalarType.Float16).cpu().bytes);
        ReadOnlySpan<Half> predictionsUncertaintyV = MemoryMarshal.Cast<byte, Half>(predictionUNC.to(ScalarType.Float16).cpu().bytes);

        ReadOnlySpan<Half> predictionsPolicy = null;
        ReadOnlySpan<Half> predictionsPolicyMasked = null;
        Tensor gatheredLegalMoveProbs = default;
        if (legalMovesIndices != null)
        {
          // The indices of legal moves were provided, therefore
          // extract only the masked legal moves (using torch gather operator).
          Tensor indices = tensor(legalMovesIndices, ScalarType.Int64, predictionPolicy.device)
                                .reshape([numPositions, TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST]);
          gatheredLegalMoveProbs = predictionPolicy.gather(1, indices);
          predictionsPolicyMasked = MemoryMarshal.Cast<byte, Half>(gatheredLegalMoveProbs.to(ScalarType.Float16).cpu().bytes);
        }
        else
        {
          predictionsPolicy = MemoryMarshal.Cast<byte, Half>(predictionPolicy.to(ScalarType.Float16).cpu().bytes);
        }

        // Create a result batch to receive results.
        // TODO: try to make this a reusable buffer,
        //       but WARNING a first attempt at this introduced serious incorrectness
        //       (maybe because the arrays were created as oversized).
        CompressedPolicyVector[] policiesToReturn = new CompressedPolicyVector[numPositions];

        // TODO: Use a ThreadStatic buffer instead.
        FP16[] w = new FP16[numPositions];
        FP16[] l = new FP16[numPositions];
        FP16[] m = new FP16[numPositions];
        FP16[] uncertaintyV = new FP16[numPositions];

        // Populate policy.
        PolicyVectorCompressedInitializerFromProbs.ProbEntry[] probs = new PolicyVectorCompressedInitializerFromProbs.ProbEntry[TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST];

        ReadOnlySpan<TPGSquareRecord> squareRecords = MemoryMarshal.Cast<byte, TPGSquareRecord>(squareBytesAll);

        bool getMGPosAtIndexWasProvided = getMGPosAtIndex != null;
        for (int i = 0; i < numPositions; i++)
        {

          if (legalMovesIndices != null)
          {
            InitPolicyProbabilities(i, probs, legalMovesIndices, predictionsPolicyMasked, policiesToReturn);
          }
          else
          {
            // ########### 1858 init
            int startIndex1858 = i * EncodedPolicyVector.POLICY_VECTOR_LENGTH;
            ReadOnlySpan<Half> spanPolicies1858 = predictionsPolicy.Slice(startIndex1858, EncodedPolicyVector.POLICY_VECTOR_LENGTH);

            if (!getMGPosAtIndexWasProvided)
            {
              ReadOnlySpan<TPGSquareRecord> thisPosSquareRecords = squareRecords.Slice(i * 64, 64);
              const bool POS_IS_WHITE_TO_MOVE = true; // TODO: Can we somehow determine this from the TPGRecord?
              //throw new Exception("Next line needs remediation (see also note in above line, we can set the color");
              Position thisPos = TPGRecord.PositionForSquares(thisPosSquareRecords, 0, POS_IS_WHITE_TO_MOVE);
              MGPosition thisPosMG = thisPos.ToMGPosition;
              getMGPosAtIndex = i => thisPosMG; // argument i is ignored deliberately, the call below will only reference current position
            }

            InitPolicyProbabilities(i, probs, getMGPosAtIndex, getMoveListAtIndex, spanPolicies1858, policiesToReturn);
          }

          w[i] = (FP16)(float)wdlProbabilitiesCPU[i * 3];
          l[i] = (FP16)(float)wdlProbabilitiesCPU[i * 3 + 2];

          float rawM = (float)predictionsMLH[i] * NetTransformer.MLH_DIVISOR;
          m[i] = (FP16)Math.Max(rawM, 0);
          uncertaintyV[i] = (FP16)(float)predictionsUncertaintyV[i];
        }

        PositionEvaluationBatch resultBatch = new PositionEvaluationBatch(IsWDL, HasM, HasUncertaintyV, numPositions, policiesToReturn,
                                                                          w, l, m, uncertaintyV, null, new TimingStats(),
                                                                          extraStats0, extraStats1, false);

        // These Tensors were created outside the DisposeScope
        // and therefore will not have been already disposed,
        // do this explicitly now to immediately free up memory.
        predictionValue.Dispose();
        predictionMLH.Dispose();
        predictionUNC.Dispose();
        predictionPolicy.Dispose();

        return resultBatch;
      }
    }




    static void InitPolicyProbabilities(int i,
                                        PolicyVectorCompressedInitializerFromProbs.ProbEntry[] probs,
                                        short[] legalMoveIndices,
                                        ReadOnlySpan<Half> spanPoliciesMasked,
                                        CompressedPolicyVector[] policiesToReturn)
    {
      // TODO: Use Span2D from performance tools NuGet package
      int baseIndex = i * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST;

      // Compute max probability (and also number of used slots).
      float maxProb = float.MinValue;
      int numUsedSlots = 0;

      for (int m = 0; m < TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST; m++)
      {
        int index = legalMoveIndices[baseIndex + m];

        // Detect possible end of moves by finding move with index zero
        // (unless appearing in first slot).
        if (m > 0 && index == 0)
        {
          break;
        }

        maxProb = MathF.Max(maxProb, (float)spanPoliciesMasked[baseIndex + m]);
        numUsedSlots++;
      }

      for (int m = 0; m < numUsedSlots; m++)
      {
        int index = legalMoveIndices[baseIndex + m];

        float probAdjusted = MathF.Exp((float)spanPoliciesMasked[baseIndex + m] - maxProb);
        probs[m] = new PolicyVectorCompressedInitializerFromProbs.ProbEntry((short)index, probAdjusted);
      }

      PolicyVectorCompressedInitializerFromProbs.InitializeFromProbsArray(ref policiesToReturn[i],
                                                                          numUsedSlots, CompressedPolicyVector.NUM_MOVE_SLOTS, probs);
    }


    public static bool MIRROR = false;

    static void InitPolicyProbabilities(int i,
                                        PolicyVectorCompressedInitializerFromProbs.ProbEntry[] probs,
                                        Func<int, MGPosition> getMGPosAtIndex,
                                        Func<int, MGMoveList> getMoveListAtIndex,
                                        ReadOnlySpan<Half> spanPolicies1858,
                                        CompressedPolicyVector[] policiesToReturn)
    {
      // Retrieve or generate legal moves in this position.
      MGMoveList movesThisPosition;
      if (getMoveListAtIndex != null)
      {
        movesThisPosition = getMoveListAtIndex(i);
      }
      else
      {
        Position thisPos = getMGPosAtIndex(i).ToPosition;

        // NN will have seen position from side to move
        bool posReversed = thisPos.SideToMove == SideType.Black;
        if (posReversed)
        {
          thisPos = thisPos.Reversed; // TODO: make more efficient
        }

        MGPosition mgPos = thisPos.ToMGPosition;
        movesThisPosition = new MGMoveList();
        MGMoveGen.GenerateMoves(mgPos, movesThisPosition);
      }

      // Determine how many legal moves we can process (truncate if they overflow our buffer).
      int numMovesToProcess = movesThisPosition.NumMovesUsed;
      if (numMovesToProcess > probs.Length)
      {
        Console.WriteLine($"Moves overflow, discarding those above {probs.Length}");
        numMovesToProcess = probs.Length;
      }

      // Compute maximum policy logit in NN array over the set of legal moves.
      float maxV1858 = float.MinValue;
      for (int j = 0; j < numMovesToProcess; j++)
      {
        EncodedMove eMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(movesThisPosition.MovesArray[j]);
        if (MIRROR) eMove = eMove.Mirrored;
        maxV1858 = MathF.Max(maxV1858, (float)spanPolicies1858[eMove.IndexNeuralNet]);
      }

      // Populate the probs array with the legal moves and their probabilities.
      for (int j = 0; j < numMovesToProcess; j++)
      {
        EncodedMove eMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(movesThisPosition.MovesArray[j]);
        if (MIRROR) eMove = eMove.Mirrored;
        float probabilityAdjusted = MathF.Exp((float)spanPolicies1858[eMove.IndexNeuralNet] - maxV1858);
        probs[j] = new PolicyVectorCompressedInitializerFromProbs.ProbEntry((short)eMove.IndexNeuralNet, probabilityAdjusted);
      }

      // Finally, initialize the policy vector for thi sposition from the probs array.
      PolicyVectorCompressedInitializerFromProbs.InitializeFromProbsArray(ref policiesToReturn[i], numMovesToProcess,
                                                                          CompressedPolicyVector.NUM_MOVE_SLOTS, probs);
    }


    /// <summary>
    /// Returns a string description of this evaluator.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<NNEvaluatorTorchsharp {EngineType} {PytorchForwardEvaluator.ToString()} on {Device} {DataType}>";
    }


    protected override void DoShutdown()
    {

    }
  }

}
