#region Using directives

using System;
using System.Diagnostics;

using Ceres.Base.DataTypes;
using Ceres.Base.Benchmarking;
using System.Runtime.InteropServices;

using TorchSharp;
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

    public NNEvaluatorTorchsharpOptions Options;



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
                                 bool lastMovePliesEnabled = false,
                                 NNEvaluatorTorchsharpOptions options = default)
      : this(engineType, new ModuleNNEvaluatorFromTorchScript(configNetExec with { EngineType = engineType},
             (NetTransformerDef)ceresTransformerNetDef),
             configNetExec.Device, configNetExec.DataType, configNetExec.UseHistory, lastMovePliesEnabled, options)
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
                                 bool lastMovePliesEnabled = false,
                                 NNEvaluatorTorchsharpOptions options = default)
    {
      ArgumentNullException.ThrowIfNull(pytorchForwardEvaluator);

      IncludeHistory = includeHistory;
      EngineType = engineType;
      Options = options;

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

    public bool hasValue2 = true; // TODO: cleanup

    public override bool HasValueSecondary => hasValue2;

    public override bool IsEquivalentTo(NNEvaluator evaluator)
    {
      NNEvaluatorTorchsharp evalTS = evaluator as NNEvaluatorTorchsharp;

      ModuleNNEvaluatorFromTorchScript evalTSModule = PytorchForwardEvaluator as ModuleNNEvaluatorFromTorchScript;
      ModuleNNEvaluatorFromTorchScript evalTSModuleOther = evalTS.PytorchForwardEvaluator as ModuleNNEvaluatorFromTorchScript;

      if (evalTS == null)
      {
        return false;
      }

      // TODO: Are there other things that need to be checked, e.g. within the ModuleNNEvaluatorFromTorchScript.
      return (DataType == evalTS.DataType && Device == evalTS.Device
          && EngineType == evalTS.EngineType
          && IncludeHistory == evalTS.IncludeHistory
          && LastMovePliesEnabled == evalTS.LastMovePliesEnabled
          && evalTSModule.TorchscriptFileName1  == evalTSModuleOther.TorchscriptFileName1
          && evalTSModule.TorchscriptFileName2 == evalTSModuleOther.TorchscriptFileName2);
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

#if DEBUG
    public static EncodedPositionWithHistory lastPosition;
    public static IPositionEvaluationBatch lastBatch; // for debugging only
#endif

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

      if (LastMovePliesEnabled && positions is not EncodedPositionBatchFlat)
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

      TPGRecordConverter.ConvertPositionsToRawSquareBytes(positions, IncludeHistory, positions.Moves, LastMovePliesEnabled,
                                                          Options == null ? 0 : Options.QNegativeBlunders,
                                                          Options == null ? 0 : Options.QPositiveBlunders,
                                                          out mgPos, out squareBytesAll, out legalMoveIndices);
#if DEBUG
      lastPosition = positions.PositionsBuffer.Span[0];
#endif

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
#if DEBUG
            Console.WriteLine(lastPosition.FinalPosition.FEN + " TB score: " + score + " " + MathF.Round(batch.GetV(0), 1));
#endif
            Console.WriteLine();
          }
        }
      }

#if DEBUG
      lastBatch = batch;
#endif
      return batch;
    }


    ISyzygyEvaluatorEngine tbEvaluator = null; // For debug code above only


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

    // Create a result batch to receive results.
    // TODO: try to make this a reusable buffer,
    //       but WARNING a first attempt at this introduced serious incorrectness
    //       (maybe because the arrays were created as oversized).
    [ThreadStatic] static CompressedPolicyVector[] policiesToReturn;

    // TODO: Use a ThreadStatic buffer instead.
    [ThreadStatic] static FP16[] w;
    [ThreadStatic] static FP16[] l;
    [ThreadStatic] static FP16[] w2;
    [ThreadStatic] static FP16[] l2;
    [ThreadStatic] static FP16[] m;
    [ThreadStatic] static FP16[] uncertaintyV;


    [ThreadStatic] static FP16[] extraStats0;
    [ThreadStatic] static FP16[] extraStats1;



    public IPositionEvaluationBatch RunEvalAndExtractResultBatch(Func<int, MGMoveList> getMoveListAtIndex,
                                                                 int numPositions,
                                                                 Func<int, MGPosition> getMGPosAtIndex,
                                                                 byte[] squareBytesAll,
                                                                 short[] legalMovesIndices = null)
    {
      if (w == null)
      {
        policiesToReturn = new CompressedPolicyVector[MAX_BATCH_SIZE];
        w = new FP16[MAX_BATCH_SIZE];
        l = new FP16[MAX_BATCH_SIZE];
        w2 = new FP16[MAX_BATCH_SIZE];
        l2 = new FP16[MAX_BATCH_SIZE];
        m = new FP16[MAX_BATCH_SIZE];
        uncertaintyV = new FP16[MAX_BATCH_SIZE];
        extraStats0 = new FP16[MAX_BATCH_SIZE];
        extraStats1 = new FP16[MAX_BATCH_SIZE];
      }

      if (false)
      {
        // Test code to show any differences in raw input compared to last call (only first position checked).
        var subBytes = squareBytesAll.AsSpan().Slice(0, 64 * TPGRecord.BYTES_PER_SQUARE_RECORD);
        var posx = TPGRecord.PositionForSquares(MemoryMarshal.Cast<byte, TPGSquareRecord>(subBytes), 0, true);
        if (lastBytes != null)
        {
          Console.WriteLine("testcompare vs first seen");
          for (int i = 0; i < 64; i++)
            for (int j = 0; j < TPGRecord.BYTES_PER_SQUARE_RECORD; j++)
            {
              if (squareBytesAll[i * TPGRecord.BYTES_PER_SQUARE_RECORD + j] != lastBytes[i * TPGRecord.BYTES_PER_SQUARE_RECORD + j])
              {
                Console.WriteLine("diff at " + i + " " + j + "  ... breaking");
                break;
              }
            }
        }
        if (lastBytes == null)
        {
          lastBytes = new byte[SQUARE_BYTES_PER_POSITION];
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
      Tensor predictionValue2;
      Tensor predictionQDeviationLower;
      Tensor predictionQDeviationUpper;

      using (no_grad())
      {
        if (false && numPositions == 2)
        {
          Console.WriteLine("refixul ");
          for (int i= 0; i < 64 * TPGRecord.BYTES_PER_SQUARE_RECORD; i++)
          {
            squareBytesAll[64 * TPGRecord.BYTES_PER_SQUARE_RECORD + i] = squareBytesAll[i];
          }
          squareBytesAll[64 * TPGRecord.BYTES_PER_SQUARE_RECORD + 1] = 33;
        }

        // Create a Tensor of bytes still on CPU.

        //        long ja = 0;
        //        for (int i=0;i<1*64* TPGRecord.BYTES_PER_SQUARE_RECORD;i++)
        //          ja+= (squareBytesAll[i] * i) % 377;
        //        Console.WriteLine(ja); System.Environment.Exit(3);

        // Move Tensor to the desired device and data type.
        Tensor cpuTensor = tensor(squareBytesAll, [numPositions, 64, TPGRecord.BYTES_PER_SQUARE_RECORD]);
        Tensor inputSquares = cpuTensor.to(Device).to(DataType);

        // *** NOTE: The following alternate methods should be faster, but actually much slower!
// best:  Tensor inputSquares = from_array(squareBytesAll, DataType, Device).reshape([numPositions, 64, TPGRecord.BYTES_PER_SQUARE_RECORD]);
//        Tensor inputSquares = tensor(squareBytesAll, [numPositions, 64, TPGRecord.BYTES_PER_SQUARE_RECORD], DataType, Device);

        // Apply scaling factor to TPG square inputs.
        inputSquares = inputSquares.div_(ByteScaled.SCALING_FACTOR);

        // Evaluate using neural net.
        (predictionValue, predictionPolicy, predictionMLH, predictionUNC, 
          predictionValue2, predictionQDeviationLower, predictionQDeviationUpper,
          _, _) = PytorchForwardEvaluator.forwardValuePolicyMLH_UNC(inputSquares, null);

        cpuTensor.Dispose();
        inputSquares.Dispose();
      }


      using (var _ = NewDisposeScope())
      {
        Span<Half> wdlProbabilitiesCPU = ExtractValueWDL(predictionValue, Options?.ValueHead1Temperature);
        Span<Half> wdl2ProbabilitiesCPU = ExtractValueWDL(predictionValue2, Options?.ValueHead2Temperature);

        if (Options?.FractionValueHead2 != 0)
        {
          float fraction1 = 1.0f - Options.FractionValueHead2;
          float fractionNonDeblundered = Options.FractionValueHead2;

          for (int i = 0; i < wdlProbabilitiesCPU.Length; i++)
          {
            float avg = (float)wdlProbabilitiesCPU[i] * fraction1
                      + (float)wdl2ProbabilitiesCPU[i] * fractionNonDeblundered;
            wdlProbabilitiesCPU[i] = (Half)avg;
          }

        }

        ReadOnlySpan<Half> predictionQDeviationLowerCPU = MemoryMarshal.Cast<byte, Half>(predictionQDeviationLower.to(ScalarType.Float16).cpu().bytes);
        ReadOnlySpan<Half> predictionQDeviationUpperCPU = MemoryMarshal.Cast<byte, Half>(predictionQDeviationUpper.to(ScalarType.Float16).cpu().bytes);

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

        // Populate policy.
        // Convert to next line to Span
        Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probs = stackalloc PolicyVectorCompressedInitializerFromProbs.ProbEntry[TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST];

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

          if (HasValueSecondary)
          {
            w2[i] = (FP16)(float)wdl2ProbabilitiesCPU[i * 3];
            l2[i] = (FP16)(float)wdl2ProbabilitiesCPU[i * 3 + 2];
            extraStats0[i] = (FP16)(float)predictionQDeviationLowerCPU[i];
            extraStats1[i] = (FP16)(float)predictionQDeviationUpperCPU[i];
          }

          float rawM = (float)predictionsMLH[i] * NetTransformer.MLH_DIVISOR;
          m[i] = (FP16)Math.Max(rawM, 0);
          uncertaintyV[i] = (FP16)(float)predictionsUncertaintyV[i];
        }

        //Console.WriteLine((w[0] - l[0]) + " " + (w2[0] - l2[0]) +
//  "  U=" + uncertaintyV[0] + "  [-" + predictionQDeviationLowerCPU[0] + " +" + predictionQDeviationUpperCPU[0] + "]");

        PositionEvaluationBatch resultBatch = new PositionEvaluationBatch(IsWDL, HasM, HasUncertaintyV, HasValueSecondary,
                                                                          numPositions, policiesToReturn,
                                                                          w, l, w2, l2, m, uncertaintyV, null, new TimingStats(),
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


    private static Span<Half> ExtractValueWDL(Tensor predictionValue, float? temperature)
    {
      // Extract logits and possibly apply temperature if specified.
      Tensor valueFloat = predictionValue.to(ScalarType.Float32);
      if (temperature.HasValue && temperature != 1)
      {
        valueFloat /= temperature;
      }

      // Subtract the max from logits and exponentiate (in Float32 to preserve accuracy during exponentiation and division).
      Tensor max_logits = torch.max(valueFloat, dim: 1, keepdim: true).values;
      Tensor exp_logits = torch.exp(valueFloat - max_logits);

      // Sum the exponentiated logits along the last dimension and use to normalize.
      Tensor sum_exp_logits = torch.sum(exp_logits, dim: 1, keepdim: true);
      Tensor wdlProbabilities = (exp_logits / sum_exp_logits);
      Span<Half> wdlProbabilitiesCPU = MemoryMarshal.Cast<byte, Half>(wdlProbabilities.to(ScalarType.Float16).cpu().bytes);
      return wdlProbabilitiesCPU;
    }


    static void InitPolicyProbabilities(int i,
                                        Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probs,
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
                                        Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probs,
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
