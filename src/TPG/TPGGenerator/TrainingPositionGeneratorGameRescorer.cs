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
using System.Diagnostics;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NNEvaluators.LC0DLL;

using CeresTrain.TrainingDataGenerator;
using Microsoft.Extensions.Options;

#endregion

namespace CeresTrain.TPG.TPGGenerator
{
  /// <summary>
  /// Implements logic to rescore a complete training game, including:
  ///   - rewrite positions with tablebase exact values
  ///   - fixup deliberate (noise) blunders injected in games so do not contaminate prior move evaluations
  ///   - fixup unintended major blunders in play so do not contaminate prior move evaluations
  ///   - calculate certain extra optional training targets (such as measures of uncertainty).
  /// </summary>
  public class TrainingPositionGeneratorGameRescorer
  {
    // Blunders due to injected noise can be definitively identified.
    // Ignore noise blunders only if very small in magnitude.
    public readonly float SUBOPTIMAL_NOISE_BLUNDER_THRESHOLD;

    // Unintended blunders are revealed by a best Q which changes
    // dramatically after making the move.
    public readonly float SUBOPTIMAL_UNINTENDED_BLUNDER_THRESHOLD = 0.15f;

    // TODO: Make these options passed in by the caller 


    const int MAX_PLY = 512;

    EncodedTrainingPositionGame thisGame;
    EncodedPositionWithHistory[] gamePositionsBufferRAW = new EncodedPositionWithHistory[MAX_PLY];
    Memory<EncodedPositionWithHistory> gamePositionsBuffer = default;
    int numPosThisGame;

    public (float w, float d, float l)[] newResultWDL = new (float, float, float)[MAX_PLY];
    public TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo[] targetSourceInfo = new TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo[MAX_PLY];
    public bool[] REJECT_POSITION_DUE_TO_POSITION_FOCUS = new bool[MAX_PLY];
    public float[] suboptimalityNoiseBlunder = new float[MAX_PLY];
    public float[] suboptimalityUnintended = new float[MAX_PLY];
    public float?[] tablebaseWL = new float?[MAX_PLY];

    public float[] forwardSumPositiveBlunders = new float[MAX_PLY];
    public float[] forwardSumNegativeBlunders = new float[MAX_PLY];

    public float[] forwardMaxSinglePositiveBlunder = new float[MAX_PLY];
    public float[] forwardMaxSingleNegativeBlunder = new float[MAX_PLY];

    public float[] forwardMinQDeviation = new float[MAX_PLY];
    public float[] forwardMaxQDeviation = new float[MAX_PLY];

    public (float w, float d, float l)[] intermediateBestWDL = new (float w, float d, float l)[MAX_PLY];
    public float[] deltaQIntermediateBestWDL = new float[MAX_PLY];

    static (float w, float d, float l) Reverse((float w, float d, float l) v) => (v.l, v.d, v.w);

    public readonly bool EmitPlySinceLastMovePerSquare;

    public TrainingPositionGeneratorGameRescorer(float noiseBlunderThreshold, float noiseBlunderUnintendedThreshold, bool emitPlySinceLastMovePerSquare)
    {
      if (noiseBlunderThreshold <= 0)
      {
        throw new ArgumentOutOfRangeException(nameof(noiseBlunderThreshold));
      }

      SUBOPTIMAL_NOISE_BLUNDER_THRESHOLD = noiseBlunderThreshold;
      SUBOPTIMAL_UNINTENDED_BLUNDER_THRESHOLD = noiseBlunderUnintendedThreshold;
      EmitPlySinceLastMovePerSquare = emitPlySinceLastMovePerSquare;

      if (EmitPlySinceLastMovePerSquare)
      {
        lastMoveIndexBySquare = new short[MAX_PLY][];
        for (int i = 0; i < MAX_PLY; i++)
        {
          lastMoveIndexBySquare[i] = new short[MAX_PLY];
        }
      }
    }

    /// <summary>
    /// Returns reference to EncodedPositionWithHistory at specified index.
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public ref readonly EncodedPositionWithHistory PositionRef(int index) => ref gamePositionsBuffer.Span[index];


    void CalcForwardBlunders()
    {
      Array.Fill(forwardSumPositiveBlunders, 0, 0, numPosThisGame);
      Array.Fill(forwardSumNegativeBlunders, 0, 0, numPosThisGame);
      Array.Fill(forwardMinQDeviation, 0, 0, numPosThisGame);
      Array.Fill(forwardMaxQDeviation, 0, 0, numPosThisGame);

      Span<EncodedPositionWithHistory> gamePositions = gamePositionsBuffer.Span;

      for (int moveIndexInGame = 0; moveIndexInGame < numPosThisGame; moveIndexInGame++)
      {
        // Get information about this position handy.
        ref readonly EncodedPositionWithHistory thisTrainingPos = ref gamePositions[moveIndexInGame];

        // Set forward cumulative suboptimalities
        // TODO: make this more efficient
        float baselineQ = thisTrainingPos.MiscInfo.InfoTraining.BestQ;
        float minQDeviation = 0;
        float maxQDeviation = 0;
        float maxPositiveBlunder = 0;
        float maxNegativeBlunder = 0;

        for (int fwdMoveIndex = moveIndexInGame; fwdMoveIndex < numPosThisGame; fwdMoveIndex++)
        {
          bool ourSide = fwdMoveIndex % 2 == moveIndexInGame % 2;
          float qOurPerspective = ourSide ? gamePositions[fwdMoveIndex].MiscInfo.InfoTraining.BestQ
                                          : -gamePositions[fwdMoveIndex].MiscInfo.InfoTraining.BestQ;

          // Possibly update min/max Q deviation. 
          float qDeviation = qOurPerspective - baselineQ;
          float qDeviationAbs = Math.Abs(qDeviation);
          if (qDeviation < 0 && qDeviationAbs > minQDeviation)
          {
            minQDeviation = qDeviationAbs;
          }
          else if (qDeviation > 0 && qDeviationAbs > maxQDeviation)
          {
            maxQDeviation = qDeviationAbs;
          }

          // Possibly update max blunders seen (positive and negative)
          float thisBlunder = suboptimalityNoiseBlunder[fwdMoveIndex];
          if (ourSide && thisBlunder > 0)
          {
            // Our side played a blunder.
            forwardSumNegativeBlunders[moveIndexInGame] += suboptimalityNoiseBlunder[fwdMoveIndex];
            if (thisBlunder > maxNegativeBlunder)
            {
              maxNegativeBlunder = thisBlunder;
            }
          }
          else if (!ourSide && thisBlunder > 0)
          {
            forwardSumPositiveBlunders[moveIndexInGame] += thisBlunder;
            if (thisBlunder > maxPositiveBlunder)
            {
              maxPositiveBlunder = thisBlunder;
            }
          }
        }

        forwardMinQDeviation[moveIndexInGame] = minQDeviation;
        forwardMaxQDeviation[moveIndexInGame] = maxQDeviation;

        forwardMaxSinglePositiveBlunder[moveIndexInGame] = maxPositiveBlunder;
        forwardMaxSingleNegativeBlunder[moveIndexInGame] = maxNegativeBlunder;
      }
    }


    public void CalcTrainWDL(TPGGeneratorOptions.DeblunderType deblunderType, bool doTBRescore, bool positionFocusEnabled)
    {
      CalcForwardBlunders();

      (float w, float d, float l) currentTrainWDL = default;

      Span<EncodedPositionWithHistory> positionsSpan = gamePositionsBuffer.Span;

      int i = numPosThisGame - 1;
      do
      {
        targetSourceInfo[i] = TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo.Training;

        ref readonly EncodedPositionWithHistory thisTrainingPos = ref positionsSpan[i];
        EncodedPositionEvalMiscInfoV6 thisInfoTraining = thisGame.PositionTrainingInfoAtIndex(i);

        TrainingPositionFocusCalculator calc = new();

        // Check for value differential focus choosing to reject this position.
        REJECT_POSITION_DUE_TO_POSITION_FOCUS[i] = false;
        if (positionFocusEnabled && new TrainingPositionFocusCalculator().CalcAcceptPosition(this, i))
        {
          REJECT_POSITION_DUE_TO_POSITION_FOCUS[i] = true;
        }

        if (i == numPosThisGame - 1)
        {
          currentTrainWDL = thisInfoTraining.ResultWDL;
        }

        // Always override current position evaluation based on tablebase value (if any).
        if (doTBRescore && tablebaseWL[i].HasValue)
        {
          currentTrainWDL = (0, 0, 0);
          if (tablebaseWL[i].Value == 1)
          {
            currentTrainWDL.w = 1;
          }
          else if (tablebaseWL[i].Value == -1)
          {
            currentTrainWDL.l = 1;
          }
          else if (tablebaseWL[i].Value == 0)
          {
            currentTrainWDL.d = 1;
          }
          else
          {
            throw new NotImplementedException("Unexpected tablebase value");
          }

          targetSourceInfo[i] = TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo.Tablebase;
        }

        // If this was a blunder (either due to noise or unintentional very bad mistake move)
        // don't let prior move evaluations be distorted by this mistake.
        // Instead just use our best available evaluation as of this point (before the mistake).
        bool foundInTB = doTBRescore && tablebaseWL[i].HasValue;
        if (deblunderType != TPGGeneratorOptions.DeblunderType.None && !foundInTB)
        {
          if (deblunderType != TPGGeneratorOptions.DeblunderType.PositionQ)
          {
            throw new NotImplementedException("Only PositionQ deblunder type supported");
          }

          if (suboptimalityNoiseBlunder[i] > SUBOPTIMAL_NOISE_BLUNDER_THRESHOLD)
          {
            numNoiseBlunders++;
            currentTrainWDL = thisInfoTraining.BestWDL;
            targetSourceInfo[i] = TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo.NoiseDeblunder;
          }
          else if (suboptimalityUnintended[i] > SUBOPTIMAL_UNINTENDED_BLUNDER_THRESHOLD)
          {
            numUnintendedBlunders++;
            currentTrainWDL = thisInfoTraining.BestWDL;
            targetSourceInfo[i] = TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo.UnintendedDeblunder;
            // SHOULD_REJECT_POSITION[i] = true;
          }
        }

        // Set the result values based on current.
        newResultWDL[i] = currentTrainWDL;

        // Try to move forward some number of plys to see an intermediate forward position.
        const int NUM_MOVES_FORWARD = 5;
        const int NUM_PLY_LOOKFORWARD = NUM_MOVES_FORWARD * 2; // must remain same side to play
        Debug.Assert(NUM_PLY_LOOKFORWARD % 2 == 0); // must remain same side to play

        int numForward = 0;
        while (numForward < NUM_PLY_LOOKFORWARD
           && i + numForward < numPosThisGame - 1
           // stop searching if saw blunder
           && targetSourceInfo[i + numForward] != TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo.UnintendedDeblunder
           && targetSourceInfo[i + numForward] != TrainingPositionWriterNonPolicyTargetInfo.TargetSourceInfo.NoiseDeblunder)
        {
          numForward++;
        }

        (float w, float d, float l) wdlBestCurrent = thisTrainingPos.MiscInfo.InfoTraining.BestWDL;
        (float w, float d, float l) wdlBestForward = positionsSpan[i + numForward].MiscInfo.InfoTraining.BestWDL;

        intermediateBestWDL[i] = numForward % 2 == 0 ? wdlBestForward : Reverse(wdlBestForward);
        float currentBestWL = wdlBestCurrent.w - wdlBestCurrent.l;
        float forwardBestWL = intermediateBestWDL[i].w - intermediateBestWDL[i].l;
        deltaQIntermediateBestWDL[i] = forwardBestWL - currentBestWL;

        // Reverse perspective to other side of current evaluation.
        currentTrainWDL = Reverse(currentTrainWDL);
      } while (i-- > 0);
    }


    public void SetGame(EncodedTrainingPositionGame game)
    {
      numTBLookup = 0;
      numTBFound = 0;
      numTBRescored = 0;

      numGamesProcessed = 0;
      numPositionsProcessed = 0;
      numUnintendedBlunders = 0;
      numNoiseBlunders = 0;

      //Memory<EncodedTrainingPosition> gamePositionsBuffer
      thisGame = game;
      numPosThisGame = game.NumPositions;

      // Copy the positions into local buffer for access.
      gamePositionsBuffer = thisGame.PositionsAll(gamePositionsBufferRAW);
      if (EmitPlySinceLastMovePerSquare)
      {
        // TODO: This is inefficient, only need to clear as
        //       many slots as there were moves in the prior game.
        for (int i = 0; i < MAX_PLY; i++)
        {
          Array.Clear(lastMoveIndexBySquare[i]);
        }
      }
    }

    // For each move game and each square as of each move,
    // the index of the move in the game on which the square was last receiving a new piece.
    // Note that these are encoded such that 0 means never moved, 
    // and (for example) a value of 1 means the square received the current piece on the first move of the game.
    internal short[][] lastMoveIndexBySquare;

    public void CalcBlundersAndTablebaseLookups(ISyzygyEvaluatorEngine tbEvaluator)
    {
      Span<EncodedPositionWithHistory> gamePositions = gamePositionsBuffer.Span;

      for (int moveIndexInGame = 0; moveIndexInGame < numPosThisGame; moveIndexInGame++)
      {
        // Get information about this position handy.
        ref readonly EncodedPositionWithHistory thisTrainingPos = ref gamePositions[moveIndexInGame];
        //        EncodedPositionEvalMiscInfoV6 thisTrainingPosInfo = thisGame.PositionTrainingInfoAtIndex(moveIndexInGame);

        SetPlySinceLastMoveInfo(moveIndexInGame, thisTrainingPos);

        ref readonly EncodedPositionEvalMiscInfoV6 thisInfoTraining = ref thisTrainingPos.MiscInfo.InfoTraining;
        float suboptimalityNoise = thisInfoTraining.BestQ - thisInfoTraining.PlayedQ;
        suboptimalityNoiseBlunder[moveIndexInGame] = suboptimalityNoise;

        if (tbEvaluator != null)
        {
          tablebaseWL[moveIndexInGame] = null;

          const int MAX_PIECES_TB = 6;
          int countPieces = thisTrainingPos.BoardsHistory.History_0.CountPieces;
          if (countPieces <= MAX_PIECES_TB)
          {
            Position thisPosition = thisTrainingPos.FinalPosition; // note: this is slow (getting FEN, string concatenation)
            float? tbRescoreValue = TablebasePositionRescoredWL(tbEvaluator, in thisPosition, gamePositions, moveIndexInGame);
            tablebaseWL[moveIndexInGame] = tbRescoreValue;
          }
        }

        float thisQ = thisInfoTraining.BestQ;
        suboptimalityUnintended[moveIndexInGame] = 0; // default assumption
        if (moveIndexInGame < numPosThisGame - 2)
        {
          //if (i > 50 && thisInfoTraining.BestD > 0.90) //MathF.Abs(thisQ) > 0.70)
          ref readonly EncodedPositionWithHistory nextTrainingPosition = ref gamePositions[moveIndexInGame + 1];

          ref readonly EncodedPositionEvalMiscInfoV6 nextInfoTraining = ref nextTrainingPosition.MiscInfo.InfoTraining;
          float nextQ = -nextInfoTraining.BestQ;
          float qChangeToNext = nextQ - thisQ;

          ref readonly EncodedPositionWithHistory nextNextTrainingPosition = ref gamePositions[moveIndexInGame + 2];
          ref readonly EncodedPositionEvalMiscInfoV6 nextNextInfoTraining = ref nextNextTrainingPosition.MiscInfo.InfoTraining;
          float nextNextQ = nextNextInfoTraining.BestQ;
          float qChangeToNextNext = nextNextQ - thisQ;
          float suboptimality = -Math.Max(qChangeToNext, qChangeToNextNext);
          if (suboptimality > 0)
          {
            suboptimalityUnintended[moveIndexInGame] = suboptimality;
          }
        }
      }
    }

    private void SetPlySinceLastMoveInfo(int moveIndexInGame, in EncodedPositionWithHistory thisTrainingPos)
    {
      if (EmitPlySinceLastMovePerSquare && moveIndexInGame > 0)
      {
        // Carry forward last move indices from last move
        // (but have to mirror).
        for (int s = 0; s < 64; s++)
        {
          Square thisSquare = new Square(s, Square.SquareIndexType.BottomToTopRightToLeft);
          lastMoveIndexBySquare[moveIndexInGame][thisSquare.Reversed.SquareIndexStartH1] = lastMoveIndexBySquare[moveIndexInGame - 1][s];
        }

        // Now update for the last move seen (note that first move is encoded as 1, not 0).
        (PieceType pieceType, Square fromSquare, Square toSquare, bool wasCastle) = thisTrainingPos.LastMoveInfoFromSideToMovePerspective();
        lastMoveIndexBySquare[moveIndexInGame][fromSquare.SquareIndexStartA1] = (short)(1 + moveIndexInGame);
        lastMoveIndexBySquare[moveIndexInGame][toSquare.SquareIndexStartA1] = (short)(1 + moveIndexInGame);
      }
    }

    public int numTBLookup = 0;
    public int numTBFound = 0;
    public int numTBRescored = 0;

    public int numGamesProcessed;
    public int numPositionsProcessed;
    public int numUnintendedBlunders;
    public int numNoiseBlunders;

    float? TablebasePositionRescoredWL(ISyzygyEvaluatorEngine tbEvaluator, in Position thisPosition, Span<EncodedPositionWithHistory> gamePositionsBuffer, int i)
    {
      // TODO: don't hardcode piece count
      if (thisPosition.PieceCount <= 6)
      {
        numTBLookup++;

        tbEvaluator.ProbeWDL(in thisPosition, out SyzygyWDLScore score, out SyzygyProbeState state);

        if (state == SyzygyProbeState.Ok)
        {
          numTBFound++;

          float? newVal = null;
          switch (score)
          {
            case SyzygyWDLScore.WDLWin:
              newVal = 1.0f;
              break;

            case SyzygyWDLScore.WDLDraw:
              newVal = 0.0f;
              break;

            case SyzygyWDLScore.WDLLoss:
              newVal = -1.0f;
              break;
          }

          if (newVal == null)
          {
            return null;
          }
          else
          {
            float startQ = gamePositionsBuffer[i].MiscInfo.InfoTraining.ResultQ;
            bool rescoreChanged = startQ != newVal;
            if (rescoreChanged)
            {
              numTBRescored++;
            }
            return newVal;
          }

        }
      }
      return null;
    }


    /// <summary>
    /// Dumps all moves in current game to console, showing various information
    /// about training data and subsequent deblunder/tablebase operations
    /// and final training WDL determined.
    /// </summary>
    public void Dump((float v1, float v2)[] extraValues = null)
    {
      const string HEADER = "      Blun+ Blun-    QDev- QDev+   Max-  Max+  Move     Source        OrgZ      NewZ  TrainQ                               W    D    L                         W    D    L";
      Console.WriteLine(HEADER);

      numGamesProcessed++;
      for (int indexPlyThisGame = 0; indexPlyThisGame < numPosThisGame; indexPlyThisGame++)
      {
        numPositionsProcessed++;

        if (indexPlyThisGame == 0)
        {
          Console.WriteLine();
        }

        // Get position information handy.
        EncodedPositionWithHistory thisPos = thisGame.PositionAtIndex(indexPlyThisGame);
        EncodedPositionEvalMiscInfoV6 thisInfoTraining = thisGame.PositionTrainingInfoAtIndex(indexPlyThisGame);

        // Set noise suboptimality.
        float suboptimalityNoise = thisInfoTraining.BestQ - thisInfoTraining.PlayedQ;
        suboptimalityNoiseBlunder[indexPlyThisGame] = suboptimalityNoise;

        string noiseBlunderStr = suboptimalityNoiseBlunder[indexPlyThisGame] > 0.01 ? $"{suboptimalityNoiseBlunder[indexPlyThisGame],5:F2}" : "     ";

        string selfBlunderStr = "        ";
        selfBlunderStr = suboptimalityUnintended[indexPlyThisGame] > SUBOPTIMAL_UNINTENDED_BLUNDER_THRESHOLD ? $"{suboptimalityUnintended[indexPlyThisGame],5:F2}" : "     ";
        //Console.WriteLine((float)numSelfBlunders / numGamesProcessed);


        Position thisPosition = thisPos.FinalPosition;
        string fen = thisPosition.FEN;
        (PieceType pieceType, Square fromSquare, Square toSquare, bool wasCastle) lastMoveInfo = thisPos.LastMoveInfoFromSideToMovePerspective();
#if NOT
// TODO: in Console.WriteLine below show an MGMove string instead of raw thisInfoTraining.PlayedIndex
        MGMove playedMoveMG = thisPos.PlayedMove;
        if (!thisPosition.ToMGPosition.IsLegalMove(playedMoveMG))
        {
          throw new Exception("TAR contained illegal played move (PlayedIndex)");
        }
#endif
        //string TB_FLAT = tablebaseWL[i].HasValue ? $"{(int)tablebaseWL[i].Value,2:N}" : "  ";

        float trainingQ = thisInfoTraining.BestWDL.w - thisInfoTraining.BestWDL.l;
        float newResultWL = newResultWDL[indexPlyThisGame].w - newResultWDL[indexPlyThisGame].l;
        float origResultWL = thisInfoTraining.ResultQ;
        float deltaEval = newResultWL - trainingQ;

        Square fromSquare = !thisPosition.IsWhite ? lastMoveInfo.fromSquare.Mirrored.Reversed : lastMoveInfo.fromSquare.Mirrored;
        Square toSquare = !thisPosition.IsWhite ? lastMoveInfo.toSquare.Mirrored.Reversed : lastMoveInfo.toSquare.Mirrored;

        string nonBestMoveChar = thisInfoTraining.NotBestMove ? "x" : " ";
        bool isLastPosition = indexPlyThisGame == numPosThisGame - 1;
        bool terminalBlunder = isLastPosition && LC0TrainingPosGeneratorFromSingleNNEval.TrainingPosWasForcedMovePossiblySeriousBlunder(in thisPos);
        string terminalBlunderChar = terminalBlunder ? "B" : " ";
        string targetSourceStr = targetSourceInfo[indexPlyThisGame].ToString();
        targetSourceStr = $"{(targetSourceStr.Length > 9 ? targetSourceStr.Substring(0, 9) : targetSourceStr.PadRight(9, ' '))}";

        string prefix = "";
        if (extraValues != null)
        {
          prefix += $"  {extraValues[indexPlyThisGame].v1,5:F2} {extraValues[indexPlyThisGame].v2,5:F2}  ";
        }

        // Add in information relating to position focus
        TrainingPositionFocusCalculator focusCalc = new TrainingPositionFocusCalculator();
        bool focusAccepted = focusCalc.CalcAcceptPosition(this, indexPlyThisGame);
        prefix += focusCalc.AcceptShortInfoString;       

        prefix += $"  {forwardSumPositiveBlunders[indexPlyThisGame],5:F2} {forwardSumNegativeBlunders[indexPlyThisGame],5:F2}  ";
        prefix += $"  {forwardMinQDeviation[indexPlyThisGame],5:F2} {forwardMaxQDeviation[indexPlyThisGame],5:F2}  ";
        prefix += $"  {forwardMaxSingleNegativeBlunder[indexPlyThisGame],5:F2} {forwardMaxSinglePositiveBlunder[indexPlyThisGame],5:F2}  ";

        Console.WriteLine($"{prefix} {indexPlyThisGame,3:N0} {nonBestMoveChar} {terminalBlunderChar} {targetSourceStr}"
                        + $"  WL {origResultWL,5:F2} --> {newResultWL,5:F2}  (bestQ= {trainingQ,5:F2} ` {deltaEval,5:F2})"
                        + $"  TGT --> {newResultWDL[indexPlyThisGame].w,4:F2} {newResultWDL[indexPlyThisGame].d,4:F2} {newResultWDL[indexPlyThisGame].l,4:F2}"
                        + $" mlh={thisInfoTraining.PliesLeft,4:N0}/{thisInfoTraining.BestM,4:N0}  "
                        + $" [fwd={intermediateBestWDL[indexPlyThisGame].w,4:F2} {intermediateBestWDL[indexPlyThisGame].d,4:F2} {intermediateBestWDL[indexPlyThisGame].l,4:F2}  delta={deltaQIntermediateBestWDL[indexPlyThisGame],5:F2}]  "
                        + $" [ns_bl={noiseBlunderStr}  slf_bl{selfBlunderStr}]  "
                        + $" {fen}  play: {thisInfoTraining.PlayedIndex}  ast: {lastMoveInfo.pieceType} {fromSquare} {toSquare} {lastMoveInfo.wasCastle}");

      }
    }

  }
}


#if EXAMPLE

const int NUM_POS_PER_SET = 4096 * 5000;

// Evaluate using an ensemble of two T60 nets (with dual GPUs)
NNEvaluator nNEvaluator = NNEvaluator.FromSpecification("LC0:j94-100@0.5,69146@0.5", "GPU:0,1");

while (true)
{
  TPGOptions options = new TPGOptions()
  {
    Description = "Rescored queenless endgames with T60 annotation",
    SourceDirectory = SoftwareManager.IsLinux ? @"/raid/train/games_tar"
                                              : @"g:\t60",// @"d:\tars\v6", //@"f:\v6",

    // Use TARs with games from only April 2021
    FilenameFilter = f => f.Contains("202104"),

    // Endgame filter to only accept positions with at most 10 positions and queenless 
    AcceptRejectAnnotater = (EncodedTrainingPosition[] gamePositions, int positionIndex, Position position)
      => position.PieceCountOfType(new Piece(SideType.White, PieceType.Queen)) == 0
                          && position.PieceCountOfType(new Piece(SideType.Black, PieceType.Queen)) == 0
                          && position.PieceCount <= 10,

    NumPositionsPerSetToGenerate = NUM_POS_PER_SET,
    RescoreWithTablebase = true,

    // Write ensembled V to unused field in training data for possible distillation.
    AnnotationNNEvaluator = nNEvaluator,
    AnnotationPostprocessor = (Position position, NNEvaluatorResult nnEvalResult, in EncodedTrainingPosition trainingPosition) =>
    {
      trainingPosition.Position.MiscInfo.SetUnusedFields(nnEvalResult.V, float.NaN);
      //Console.WriteLine($"Saw nnV={nnEvalResult.V} fileQ=  {trainingPosition.Position.MiscInfo.InfoTraining.ResultQ} pos: {position.FEN}");
    },

    //            TargetFileName = @$"d:\train\tpg\pos{DateTime.Now.Ticks % 100000}.gz",
    TargetFileNameBase = SoftwareManager.IsLinux ? @$"/raid/train/tpg/pos{DateTime.Now.Ticks % 100000}"
                                             : @$"d:\train\tpg\pos{DateTime.Now.Ticks % 100000}",
    NumThreads = 24
  };

  TrainingPositionGenerator tpg = new(options);

  tpg.RunGeneratorLoop();

#endif