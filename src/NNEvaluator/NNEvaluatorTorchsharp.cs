﻿#region Using directives

using System;
using System.Diagnostics;
using System.Linq;

using System.Collections.Generic;
using System.Runtime.InteropServices;

using SharpCompress;

using Ceres.Base.DataTypes;
using Ceres.Base.Benchmarking;

using TorchSharp;
using static TorchSharp.torch;

using Ceres.Base.DataType;
using Ceres.Base.Misc;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.NNEvaluators.Ceres;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.MoveGen;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.NNEvaluators.LC0DLL;

using CeresTrain.Networks.Transformer;
using CeresTrain.Trainer;
using CeresTrain.TrainCommands;
using CeresTrain.Utils;

using static TorchSharp.torch.nn;

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
    public IModuleNNEvaluator PytorchForwardEvaluator;

    /// <summary>
    /// Returns the 
    /// </summary>
    public jit.ScriptModule<Tensor, Tensor, Tensor[]> Module => PytorchForwardEvaluator.Module;

    /// <summary>
    /// Options for the evaluator.
    /// </summary>
    public NNEvaluatorOptionsCeres OptionsTorchsharp
    {
      get => (NNEvaluatorOptionsCeres)Options;
    }

    /// <summary>
    /// Returns information about the evaluator.
    /// </summary>
    public override EvaluatorInfo Info =>  getNumModelParams == null ? null : new EvaluatorInfo(getNumModelParams());  


    /// <summary>
    /// Function to get the number of model parameters.
    /// Only called if required to avoid overhead of computing this value.
    /// </summary>
    Func<long> getNumModelParams = null;

    bool hasAction => OptionsTorchsharp.UseAction;


    /// <summary
    /// Constructor.
    /// </summary>
    /// <param name="engineType"></param>
    /// <param name="ceresTransformerNetDef"></param>
    /// <param name="configNetExec"></param>
    /// <param name="lastMovePliesEnabled"></param>
    public NNEvaluatorTorchsharp(NNEvaluatorInferenceEngineType engineType, 
                                 ConfigNetExecution configNetExec,
                                 Device device, ScalarType dataType,
                                 bool lastMovePliesEnabled = false,
                                 NNEvaluatorOptionsCeres options = default,
                                 NetTransformerDef netTransformerDef = default)
      : this(engineType, 
            new ModuleNNEvaluatorFromTorchScript(configNetExec with { EngineType = engineType},
                                                 device, dataType, options.UsePriorState, netTransformerDef),
                                                 device, dataType,
                                                 true, lastMovePliesEnabled, options, netTransformerDef)
    {
      getNumModelParams = () => TorchscriptUtils.NumParameters(configNetExec.SaveNetwork1FileName);
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
                                 NNEvaluatorOptionsCeres options = default,
                                 NetTransformerDef netTransformerDef = default)
    {
      OID = DateTime.Now.Ticks.GetHashCode();

      ArgumentNullException.ThrowIfNull(options);
      ArgumentNullException.ThrowIfNull(pytorchForwardEvaluator);

      IncludeHistory = includeHistory;
      EngineType = engineType;
      Options = options;

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
    public override bool HasUncertaintyP => true;
    public override bool HasAction => hasAction;

    public override bool HasState => OptionsTorchsharp.UsePriorState;

    public bool hasValue2 = true; // TODO: cleanup

    public override bool HasValueSecondary => hasValue2;

    int OID;


    public void ReplaceValue1Head(Module<Tensor,Tensor> moduleValueHead)
    {
      NetTransformer transformer = (PytorchForwardEvaluator as ModuleNNEvaluatorFromTorchScript).CeresNet as NetTransformer;
      transformer.layerValueHead = moduleValueHead;
    }


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
          && HasAction == evalTS.HasAction
          && evalTSModule.TorchscriptFileName1 == evalTSModuleOther.TorchscriptFileName1
          && evalTSModule.TorchscriptFileName2 == evalTSModuleOther.TorchscriptFileName2
          && Options == evalTS.Options
          );
    }


    /// <summary>
    /// Maximum supported batch size.
    /// </summary>
    public const int MAX_BATCH_SIZE = 1024;

    public override int MaxBatchSize { get; } = MAX_BATCH_SIZE;

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

    short[] legalMoveIndicesBuffer = new short[MAX_BATCH_SIZE * TPGRecord.MAX_MOVES];


    public IPositionEvaluationBatch Evaluate(ReadOnlySpan<TPGRecord> tpgRecords)
    {
      if (tpgRecords.Length > MAX_BATCH_SIZE)
      {
        throw new Exception($"NNEvaluatorTorchsharp: requested batch size of {tpgRecords.Length} exceeds maximum supported of {MAX_BATCH_SIZE}");
      }

      Func<int, MGMoveList> getMoveListAtIndex = (i) =>
      {
        MGMoveList moveList = new MGMoveList();
        MGMoveGen.GenerateMoves(in mgPos[i], moveList);
        return moveList;
      };

      // TODO: This is memory/compute inefficient
      short[] legalMoveIndices = new short[tpgRecords.Length * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST];
      for (int i = 0; i < tpgRecords.Length; i++)
      {
        // Extract as bytes.
        tpgRecords[i].CopySquares(squareBytesAll, i * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD);
        mgPos[i] = tpgRecords[i].FinalPosition.ToMGPosition;

        TPGRecordMovesExtractor.ExtractLegalMoveIndicesForIndex(tpgRecords, default, legalMoveIndices, i);
      }

      // TODO: The state get Func in next line is null, there is currently no way to pass in state to this method. Enhance someday.
      return RunEvalAndExtractResultBatch(null, getMoveListAtIndex, tpgRecords.Length, i => mgPos[i], squareBytesAll,
      legalMovesIndices: legalMoveIndices);

    }

    static bool haveInitialized = false;
    static bool haveWarned = false;
    static bool haveWarned1 = false;
    bool firstTime = true;
    public EncodedPositionWithHistory lastPosition;
    public static EncodedPositionWithHistory lastPositionStatic;

#if DEBUG
    public IPositionEvaluationBatch lastBatch; // for debugging only
    public static IPositionEvaluationBatch lastBatchStatic; // for debugging only
#endif

    //    Dictionary<string, float[]> cachedPriorHistory = new Dictionary<string, float[]>();

    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      firstTime = false;
//      bool equal = !lastPosition.BoardsHistory.BoardAtIndex(0).IsEmpty &&
//        positions.PositionsBuffer.Span[0].HistoryPosition(1).PiecesEqual(lastPosition.HistoryPosition(0));
//      Console.WriteLine("equalxxx " + equal);


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

      TPGRecordConverter.ConvertPositionsToRawSquareBytes(positions, IncludeHistory, positions.Moves, LastMovePliesEnabled,
                                                          OptionsTorchsharp.QNegativeBlunders, OptionsTorchsharp.QPositiveBlunders,
                                                          out mgPos, out squareBytesAll, legalMoveIndicesBuffer);
#if DEBUG
      lastPosition = lastPositionStatic = positions.PositionsBuffer.Span[0];

#endif

      IPositionEvaluationBatch batch = RunEvalAndExtractResultBatch(i => positions.States.Length == 0 ? default : positions.States.Span[i], 
                                                                    i => positions.Moves.Span[i], 
                                                                    positions.NumPos,
                                                                    i => mgPos[i], squareBytesAll, legalMoveIndicesBuffer);
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
      lastBatch = lastBatchStatic = batch;
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

      if (HasState)
      {
        throw new NotImplementedException("Need remediation for RunEvalAndExtractResultBatch below");

      }
      // Extract the square records (as byte arrays) from each of the TPGRecords.
      TPGRecord[] recs = (TPGRecord[])positionsNativeInput;
      byte[] rawBytes = new byte[numPositions * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD];
      for (int i = 0; i < numPositions; i++)
      {
        recs[i].CopySquares(rawBytes, i * 64 * TPGRecord.BYTES_PER_SQUARE_RECORD);
      }
      IPositionEvaluationBatch ret = RunEvalAndExtractResultBatch(null, null, numPositions, null, rawBytes, null);

      return ret;
    }


    static readonly int SQUARE_BYTES_PER_POSITION = 64 * TPGRecord.BYTES_PER_SQUARE_RECORD;

    static byte[] lastBytes = null;

    // Create a result batch to receive results.
    // TODO: try to make this a reusable buffer,
    //       but WARNING a first attempt at this introduced serious incorrectness
    //       (maybe because the arrays were created as oversized).
    [ThreadStatic] static CompressedPolicyVector[] policiesToReturn;

    [ThreadStatic] static CompressedActionVector[] actionsToReturn;

    // TODO: Use a ThreadStatic buffer instead.
    [ThreadStatic] static FP16[] w;
    [ThreadStatic] static FP16[] l;
    [ThreadStatic] static FP16[] w2;
    [ThreadStatic] static FP16[] l2;
    [ThreadStatic] static FP16[] m;
    [ThreadStatic] static FP16[] uncertaintyV;
    [ThreadStatic] static FP16[] uncertaintyP;
    [ThreadStatic] static Half[] state;


    [ThreadStatic] static FP16[] extraStats0;
    [ThreadStatic] static FP16[] extraStats1;


    /// <summary>
    /// Returns exponentiated logits (after subtracting max to avoid overflow) as a Span of floats.
    /// Note that any unused slots will have been filled with copies of the last element.
    /// Therefore it is not correct to apply summation here (but max operator is safe).
    /// </summary>
    /// <param name="policies"></param>
    /// <param name="temperatureBase"></param>
    /// <returns></returns>
    private static Span<Half> ExtractExponentiatedPolicyProbabilities(Tensor policies, float temperatureBase, 
                                                                       Tensor policyUncertainties, 
                                                                       float policyUncertaintyMultiplier = 0)
    {
      Debug.Assert(temperatureBase > 0);

      // Extract logits and possibly apply temperature if specified.
      Tensor valueFloat = policies.to(ScalarType.Float32);
      if (policyUncertaintyMultiplier > 0)
      {
        const float MAX_POLICY_UNCERAINTY = 0.20f;
        var tensorxx = (temperatureBase + torch.min(MAX_POLICY_UNCERAINTY, policyUncertainties) * policyUncertaintyMultiplier * 1f); ;
//        Console.WriteLine(tensorxx.shape);
        valueFloat = valueFloat / (temperatureBase + torch.min(MAX_POLICY_UNCERAINTY, policyUncertainties) * policyUncertaintyMultiplier * 1f);
      }
      else
      {
        if (temperatureBase != 1)
        {
          valueFloat /= temperatureBase;
        }
      }

      // Subtract the max from logits and exponentiate (in Float32 to preserve accuracy during exponentiation and division).
      // TODO: consider doing this directly with torch.logit.
      Tensor max_logits = torch.max(valueFloat, dim: 1, keepdim: true).values;
      Tensor exp_logits = torch.exp(valueFloat - max_logits);

      return MemoryMarshal.Cast<byte, Half>(exp_logits.to(ScalarType.Float16).cpu().bytes);
    }


      
  public IPositionEvaluationBatch RunEvalAndExtractResultBatch(Func<int, Half[]> getState,
                                                               Func<int, MGMoveList> getMoveListAtIndex,
                                                               int numPositions,
                                                               Func<int, MGPosition> getMGPosAtIndex,
                                                               byte[] squareBytesAll,
                                                               short[] legalMovesIndices = null)
    {
      if (w == null)
      {
        policiesToReturn = new CompressedPolicyVector[MAX_BATCH_SIZE];
        actionsToReturn = new CompressedActionVector[MAX_BATCH_SIZE];
        w = new FP16[MAX_BATCH_SIZE];
        l = new FP16[MAX_BATCH_SIZE];
        w2 = new FP16[MAX_BATCH_SIZE];
        l2 = new FP16[MAX_BATCH_SIZE];
        m = new FP16[MAX_BATCH_SIZE];
        uncertaintyV = new FP16[MAX_BATCH_SIZE];
        uncertaintyP = new FP16[MAX_BATCH_SIZE];
        extraStats0 = new FP16[MAX_BATCH_SIZE];
        extraStats1 = new FP16[MAX_BATCH_SIZE];
        if (OptionsTorchsharp.UsePriorState && getState != null)
        {
          state = new Half[MAX_BATCH_SIZE * 64 * 32]; // TODO: improve, hardcoded to largest expected state size (4k each position)
        }

      }

      if (squareBytesAll.Length > numPositions * SQUARE_BYTES_PER_POSITION)
      {
        // TODO: could we create tensor belwo from a span directly instead?
        Array.Resize(ref squareBytesAll, numPositions * SQUARE_BYTES_PER_POSITION);
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

      Tensor actions;
      Tensor boardStateInput = default;
      Tensor boardStateOutput;
      Tensor actionUncertainty;
      Tensor uncertaintyPolicy;

      using (no_grad())
      {
        // Move Tensor to the desired device and data type.
        Tensor inputSquares = from_array(squareBytesAll, DataType, Device).reshape([numPositions, 64, TPGRecord.BYTES_PER_SQUARE_RECORD]);

        // Apply scaling factor to TPG square inputs.
        inputSquares = inputSquares.div_(ByteScaled.SCALING_FACTOR);

#if NOT
        while (true)
        {
          using (new TimingBlock("xx"))
          {
            for (int i = 0; i < 100; i++)
            {
              var x = PytorchForwardEvaluator.forwardValuePolicyMLH_UNC((inputSquares, null));
            }
          }
        }
#endif


        if (OptionsTorchsharp.UsePriorState && getState != null && getState(0) != null)
        {
          // Allocate buffer to hold values across states for all positions.
          int stateLength = getState(0).Length;
          Half[] allStates;

          if (numPositions == 1)
          {
            allStates = getState(0);
          }
          else
          {
            allStates = new Half[numPositions * stateLength];

            // Copy each position's state into the buffer.
            for (int i = 0; i < numPositions; i++)
            {
              Half[] thisState = getState(i);
              Array.Copy(thisState, 0, allStates, i * stateLength, stateLength);
            }
          }

          // Set boardState to a tensor created from allStates
          boardStateInput = from_array(allStates, DataType, Device).reshape([numPositions, stateLength]);
        }
        else
        {
          if (HasState)
          { 
            boardStateInput = torch.zeros([numPositions, 256], device:Device, dtype:DataType); // HARDCODEFIX
          }
        }

        (predictionPolicy, predictionValue, predictionMLH, predictionUNC, 
        predictionValue2, predictionQDeviationLower, predictionQDeviationUpper, uncertaintyPolicy,
        actions, boardStateOutput, actionUncertainty,
        _, _) = PytorchForwardEvaluator.forwardValuePolicyMLH_UNC((inputSquares, boardStateInput));

#if NOT
        if (Options.UsePriorState)
{
  var evalNoState = PytorchForwardEvaluator.forwardValuePolicyMLH_UNC((inputSquares, torch.zeros([numPositions, 256], device: Device, dtype: DataType)));
  predictionPolicy = evalNoState.policy;
}
#endif
        if (!OptionsTorchsharp.UsePriorState)
        {
          boardStateOutput?.Dispose();
          boardStateOutput = null;
        } 

        inputSquares.Dispose();
        boardStateInput?.Dispose();
      }


      using (var _ = NewDisposeScope())
      {
        // Cast data to desired C# data type and transfer to CPU.
        // Then get ReadOnlySpans over the underlying data of the tensors.
        // Note: Because this all happens within an DisposeScope, the tensors will be kept alive (not disposed)
        //       for the duration of the block therefore the Spans (which reference underlying Tensor memory at a fixed location)
        //       will be valid for the duration of the block.
        //        ReadOnlySpan<Half> predictionsValue = MemoryMarshal.Cast<byte, Half>(predictionValue.to(ScalarType.Float16).cpu().bytes);
        ReadOnlySpan<Half> predictionsMLH = ((object)predictionMLH) == null ? default: MemoryMarshal.Cast<byte, Half>(predictionMLH.to(ScalarType.Float16).cpu().bytes);
        ReadOnlySpan<Half> predictionsUncertaintyV = MemoryMarshal.Cast<byte, Half>(predictionUNC.to(ScalarType.Float16).cpu().bytes);

        ReadOnlySpan<Half> predictionQDeviationLowerCPU = MemoryMarshal.Cast<byte, Half>(predictionQDeviationLower.to(ScalarType.Float16).cpu().bytes);
        ReadOnlySpan<Half> predictionQDeviationUpperCPU = MemoryMarshal.Cast<byte, Half>(predictionQDeviationUpper.to(ScalarType.Float16).cpu().bytes);

        if (!haveWarned1)
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Red, "NNEvaluatorTorchsharp uses old code for processing Options,"
                                                        + "should instead leverage shared code now present in PositionEvaluationBatch");
          haveWarned1 = true;
        }
        Span<Half> wdlProbabilitiesCPU = ExtractValueWDL(predictionValue, Options.ValueHead1Temperature);
        Span<Half> wdl2ProbabilitiesCPU = ExtractValueWDL(predictionValue2, Options.ValueHead2Temperature, null, null);

//                                                          Options.UseValueTemperature ? predictionQDeviationLower : null,
//                                                          Options.UseValueTemperature ? predictionQDeviationUpper : null);

        float fraction1 = 1.0f - Options.FractionValueHead2;
        float fractionNonDeblundered = Options.FractionValueHead2;


        for (int i = 0; i < wdl2ProbabilitiesCPU.Length / 3; i++)
        {
          // ****** TEMPORARY ******
          // Stash value2 raw (before modified below) into MLH temporarily since MLH rarely used and we need to preserve MLH.
          m[i] = (FP16)((float)wdl2ProbabilitiesCPU[i * 3] - (float)wdl2ProbabilitiesCPU[i * 3 + 2]);
          // ***********************
        }


          for (int i = 0; i < wdlProbabilitiesCPU.Length; i++)
          {
            float avg = (float)wdlProbabilitiesCPU[i] * fraction1
                      + (float)wdl2ProbabilitiesCPU[i] * fractionNonDeblundered;
            wdlProbabilitiesCPU[i] = (Half)avg;
          }
#if NOT
        static float WtdPowerMean(float a, float b, float w1, float w2, float p)
        {
          float sum = w1 * MathF.Pow(a, p)
                    + w2 * MathF.Pow(b, p);
          sum /= (w1 + w2);
          return MathF.Pow(sum, 1.0f / p);
        }


        // Geometric mean is value as PowerMean appraches with order 0.
        // Avoid need of special-case logic, map to a value very close to 0.
        float pToUse = Options.ValueHeadAveragePowerMeanOrder == 0 ? 0.001f
                                                                       : Options.ValueHeadAveragePowerMeanOrder;

            float w = WtdPowerMean((float)wdlProbabilitiesCPU[i], (float)wdl2ProbabilitiesCPU[i], fraction1, fractionNonDeblundered, pToUse);
            float d = WtdPowerMean((float)wdlProbabilitiesCPU[i + 1], (float)wdl2ProbabilitiesCPU[i + 1], fraction1, fractionNonDeblundered, pToUse);
            float l = WtdPowerMean((float)wdlProbabilitiesCPU[i + 2], (float)wdl2ProbabilitiesCPU[i + 2], fraction1, fractionNonDeblundered, pToUse);

            float sum = w + d + l;

            wdlProbabilitiesCPU[i] = (Half)(w / sum);
            wdlProbabilitiesCPU[i + 1] = (Half)(d / sum);
            wdlProbabilitiesCPU[i + 2] = (Half)(l / sum);
#endif


        //ReadOnlySpan<Half> predictionsPolicy = null;
        ReadOnlySpan<Half> predictionsPolicyMasked = null;
        Tensor gatheredLegalMoveProbs = default;

        if (legalMovesIndices != null)
        {
          // The indices of legal moves were provided, therefore
          // extract only the masked legal moves (using torch gather operator).
          
          // Create an array of legal move indices exactly sized for this batch.
          // TODO: this is inefficent, ideally want a Torchsharp API 
          short[] legalMoveIndicesSlice = new short[numPositions * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST];
          Array.Copy(legalMovesIndices, legalMoveIndicesSlice, legalMoveIndicesSlice.Length);
          Tensor indices = from_array(legalMoveIndicesSlice, ScalarType.Int64, predictionPolicy.device)
                                .reshape([numPositions, TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST]);
          gatheredLegalMoveProbs = predictionPolicy.gather(1, indices);

//ReadOnlySpan<Half> policiesRaw = MemoryMarshal.Cast<byte, Half>(gatheredLegalMoveProbs.to(ScalarType.Float16).cpu().bytes);

          // TODO: possibly someday apply temperature directly here rather than later and more slowly in C#
          predictionsPolicyMasked = ExtractExponentiatedPolicyProbabilities(gatheredLegalMoveProbs, Options.PolicyTemperature,
                                                                            predictionQDeviationUpper, Options.PolicyUncertaintyTemperatureScalingFactor);
        }
        else
        {
          throw new NotImplementedException();
        }

        // Create a result batch to receive results.
        // TODO: try to make this a reusable buffer,
        //       but WARNING a first attempt at this introduced serious incorrectness
        //       (maybe because the arrays were created as oversized).

        // Populate policy.
        // Convert to next line to Span
        Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probs = stackalloc PolicyVectorCompressedInitializerFromProbs.ProbEntry[TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST];

        ReadOnlySpan<TPGSquareRecord> squareRecords = MemoryMarshal.Cast<byte, TPGSquareRecord>(squareBytesAll);

        Half[] actionsSpan = null;
        if (OptionsTorchsharp.UseAction &&  (object)actions != null)
        {
          const float TEMPERATURE = 1.0f;
          if (TEMPERATURE != 1)
          {
            actions /= TEMPERATURE;
          }
          
          actions = torch.nn.functional.softmax(actions, -1);
          actionsSpan = MemoryMarshal.Cast<byte, Half>(actions.to(ScalarType.Float16).cpu().bytes).ToArray();
        }

        bool getMGPosAtIndexWasProvided = getMGPosAtIndex != null;
        for (int i = 0; i < numPositions; i++)
        {

          if (legalMovesIndices != null)
          {
            InitPolicyAndActionProbabilities(i, probs, 
                                             getMGPosAtIndexWasProvided ? getMGPosAtIndex(i).SideToMove : SideType.White,
                                             legalMovesIndices, predictionsPolicyMasked, 
                                             policiesToReturn, actionsSpan, actionsToReturn, hasAction);
          }
          else
          {
            throw new Exception("this code should be retested");
#if NOT
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
#endif
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

// *** MLH DISABLED ***
//          float rawM = (float)predictionsMLH[i] * NetTransformer.MLH_DIVISOR;
//          m[i] = (FP16)Math.Max(rawM, 0);
          uncertaintyV[i] = (FP16)(float)predictionsUncertaintyV[i];
          uncertaintyP[i] = FP16.NaN; //(FP16)(float)predictionsUncertaintyP[i];
        }

        Half[][] states = null;
        if ((object)boardStateOutput != null)
        {
          // Network returned state information, store in the states (for inclusion in batch result).
          states = new Half[numPositions][];
          Tensor outputBoardState = boardStateOutput.clone().DetachFromDisposeScope();
          state = MemoryMarshal.Cast<byte, Half>(outputBoardState.to(ScalarType.Float16).cpu().bytes).ToArray();
          int stateSize = state.Length / numPositions;
          for (int i=0;i<numPositions;i++)
          {
            states[i] = state.AsSpan(i * stateSize, stateSize).ToArray();
          } 
        }
        PositionEvaluationBatch resultBatch = new PositionEvaluationBatch(IsWDL, HasM, HasUncertaintyV, HasUncertaintyP,
                                                                          HasAction, HasValueSecondary, HasState,
                                                                          numPositions, policiesToReturn, actionsToReturn,
                                                                          w, l, w2, l2, m, uncertaintyV, uncertaintyP, states, null, new TimingStats(),
                                                                          extraStats0, extraStats1, false);

        if (OptionsTorchsharp.UseAction)
        {
//          resultBatch.RewriteWDLToBlendedValueAction();
        }


        // These Tensors were created outside the DisposeScope
        // and therefore will not have been already disposed,
        // do this explicitly now to immediately free up memory.
        predictionValue.Dispose();
        predictionMLH?.Dispose();
        predictionUNC.Dispose();
        predictionPolicy.Dispose();

        return resultBatch;
      }
    }


    static float DoShrinkExtremes(float value, float start = 0.70f, float max = 0.85f)
    {

      if (Math.Abs(value) <= start)
      {
        // Identity range
        return value;
      }
      else if (Math.Abs(value) > 1f)
      {
        // Clamping range
        return Math.Sign(value) * 1f;
      }
      else
      {
        // Compression range

        // Calculate slope
        float m = (max - start) / (1f - start);

        // Use the line equation y = mx + b to calculate the new value, where b is the y-intercept
        float b = max - m * 1f; // y = mx + b => b = y - mx
        float compressedValue = m * Math.Abs(value) + b;

        // Preserve the sign of the original value
        return Math.Sign(value) * compressedValue;
      }
    }


    private static Span<Half> ExtractValueWDL(Tensor predictionValue, float? temperature, 
                                              Tensor valueTemperatures = null,
                                              Tensor uncertaintyTemperature = null)
    {
      // Extract logits and possibly apply temperature if specified.
      Tensor valueFloat = predictionValue.to(ScalarType.Float32);
      if (valueTemperatures is not null)
      {
        throw new NotImplementedException();
        temperature = temperature ?? 1.0f;

        Tensor t = temperature + 1.25 * valueTemperatures + 1 * uncertaintyTemperature;// torch.full_like(valueTemperatures, temperature);
        //          t = torch.where(uncertaintyAdjustments < 0.08, torch.tensor(temperature.Value - 0.4f), t);
        //          t = torch.where(uncertaintyAdjustments > 0.20, torch.tensor(temperature.Value + 0.6f), t);

        //t = t + 4 * (uncertaintyAdjustments - 0.1);

        valueFloat /= t;
      }
      else if (temperature.HasValue && temperature != 1)
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

      if (false)
      {
        for (int i = 0; i < wdlProbabilitiesCPU.Length; i += 3)
        {
          float TEMPERATURE = 2.2f;
          (float, float, float) redo = Rebuild(TEMPERATURE,
            (float)wdlProbabilitiesCPU[i], (float)wdlProbabilitiesCPU[i + 1],
            (float)wdlProbabilitiesCPU[i + 2]);

          wdlProbabilitiesCPU[i] = (Half)redo.Item1;
          wdlProbabilitiesCPU[i + 1] = (Half)redo.Item2;
          wdlProbabilitiesCPU[i + 2] = (Half)redo.Item3;
        }
      }

      return wdlProbabilitiesCPU;
    }

    static (float, float, float) Rebuild(float temperature, float winPRaw, float drawPRaw, float lossPRaw)
    {

      if (winPRaw < 0.5 && lossPRaw < 0.5)
      {
        return (winPRaw, drawPRaw, lossPRaw);
      } 

      (float winPRawLogit, float drawPRawLogit, float lossPRawLogit) = (MathF.Log(winPRaw) / temperature, MathF.Log(drawPRaw) / temperature, MathF.Log(lossPRaw) / temperature);
      (float winPAdj, float drawPAdj, float lossPAdj) = (MathF.Exp(winPRawLogit), MathF.Exp(drawPRawLogit), MathF.Exp(lossPRawLogit));
      float sum = winPAdj + drawPAdj + lossPAdj;

      (winPAdj, drawPAdj, lossPAdj) = (winPAdj / sum, drawPAdj / sum, lossPAdj / sum);

      if (winPAdj > drawPAdj && winPAdj > lossPAdj)
      {
        float toDistribute = winPRaw - winPAdj;

        drawPAdj += 0.80f * toDistribute;
        lossPAdj += 0.20f * toDistribute;
      }
      else if (lossPAdj > winPAdj && lossPAdj > drawPAdj)
      {
        float toDistribute = lossPRaw - lossPAdj;

        drawPAdj += 0.80f * toDistribute;
        winPAdj += 0.20f * toDistribute;
      }
      else
      {
      }
 
      return (winPAdj, drawPAdj, lossPAdj); 
    }


    internal static void Smooth(ref CompressedActionVector actions, 
                                in CompressedPolicyVector policies, 
                                float maxDistance, 
                                float smoothedValueWeight)
    {
      // TODO: make this more efficient
      float[] probs = policies.ProbabilitySummary().AsEnumerable().Select(x => x.Probability).ToArray();
      int numMoves = probs.Length;

      CompressedActionVector tempAValue = default;

      for (int i = 0; i < numMoves; i++)
      {
        float sumW = 0;
        float sumL = 0;
        int count = 0;

        // Find elements within the specified distance in PROBS
        for (int j = 0; j < numMoves; j++)
        {
          if (Math.Abs(probs[i] - probs[j]) < maxDistance)
          {
            sumW += (float)actions[j].W;
            sumL += (float)actions[j].L;
            count++;
          }
        }

        if (count > 1) // If there was some action value other than ourself
        {
          float averageW = sumW / count;
          float averageL = sumL / count;

          // Calculate the new value using the weighted average formula
          tempAValue[i].W = (Half)((1 - smoothedValueWeight) * (float)actions[i].W + smoothedValueWeight * averageW);
          tempAValue[i].L = (Half)((1 - smoothedValueWeight) * (float)actions[i].L + smoothedValueWeight * averageL);
        }
        else
        {
          tempAValue[i] = actions[i]; // no change
        }
      }

      // Copy the temporary array back to AVALUE
      for (int i = 0; i < numMoves; i++)
      {
        actions[i] = tempAValue[i];
      }     
    }

    
    static void InitPolicyAndActionProbabilities(int i,
                                                 Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probs,
                                                 SideType side,
                                                 short[] legalMoveIndices,
                                                 ReadOnlySpan<Half> spanPoliciesMaskedAndExponentiated,
                                                 CompressedPolicyVector[] policiesToReturn,
                                                 Span<Half> actionValues,
                                                 CompressedActionVector[] actionsToReturn,
                                                 bool hasActions)
    {
      // TODO: Use Span2D from performance tools NuGet package
      int baseIndex = i * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST;

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

        if (!actionValues.IsEmpty
          && actionValues.Length >= 1858 * 3 // Temporary addition check to recognize networks with older format
          )
        {
          int actionBaseIndex = i * 1858 * 3 + index * 3;
          Half w = actionValues[actionBaseIndex];
          Half l = actionValues[actionBaseIndex + 2];
          Debug.Assert(w >= Half.Zero && l >= Half.Zero); // expected already converted from logits to probabilities

          actionsToReturn[i][m] = (w, l);
        }

        //  float probAdjusted = MathF.Exp((float)spanPoliciesMasked[baseIndex + m] - maxProb);
        float thisProb = (float)spanPoliciesMaskedAndExponentiated[baseIndex + m];  
        probs[m] = new PolicyVectorCompressedInitializerFromProbs.ProbEntry((short)index, thisProb);
        numUsedSlots++;
      }

      PolicyVectorCompressedInitializerFromProbs.InitializeFromProbsArray(ref policiesToReturn[i],
                                                                          ref actionsToReturn[i], 
                                                                          side,
                                                                          hasActions,
                                                                          numUsedSlots, CompressedPolicyVector.NUM_MOVE_SLOTS, probs);
    }


    public static bool MIRROR = false;

    static void InitPolicyProbabilities(int i,
                                        Span<PolicyVectorCompressedInitializerFromProbs.ProbEntry> probs,
                                        Func<int, MGPosition> getMGPosAtIndex,
                                        Func<int, MGMoveList> getMoveListAtIndex,
                                        ReadOnlySpan<Half> spanPolicies1858,
                                        CompressedPolicyVector[] policiesToReturn,
                                        CompressedActionVector[] actionsToReturn,
                                        bool hasAction)
    {
      SideType side;

      // Retrieve or generate legal moves in this position.
      MGMoveList movesThisPosition;
      if (getMoveListAtIndex != null)
      {
        movesThisPosition = getMoveListAtIndex(i);

        side = getMGPosAtIndex == null ? SideType.White : getMGPosAtIndex(i).SideToMove;
      }
      else
      {
        Position thisPos = getMGPosAtIndex(i).ToPosition;
        side = thisPos.SideToMove;

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
      PolicyVectorCompressedInitializerFromProbs.InitializeFromProbsArray(ref policiesToReturn[i], 
                                                                          ref actionsToReturn[i],
                                                                          side,
                                                                          hasAction,
                                                                          numMovesToProcess,
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
