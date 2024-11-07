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
using Ceres.Chess;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Features.GameEngines;
using Ceres.Chess.Positions;
using System.Collections.Generic;
using System.Threading.Tasks;
using Ceres.Chess.UserSettings;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.GameEngines;
using Ceres.Base.DataTypes;
using Ceres.MCTS.Evaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators;

#endregion 

namespace CeresTrain.Examples
{
  public static class SFEvaluatorPool
  {
    public static void InstallEvaluatorNNSFRewriter()
    {
      LeafEvaluatorNN.RewriteBatchQFunc = RewriteBatchQ;
    }

    const bool SF_WITH_TABLEBASE = true;
    const int SF_NUM_THREADS = 1;

    static readonly object creatorObj = new();

    const int MAX_SF_ENGINES = 15;
    const int MAX_BT4_THREADS = 1;
    static ObjectPool<object> sfEnginePool = new(() => CreateSFEngine(SF_NUM_THREADS, SF_WITH_TABLEBASE), MAX_SF_ENGINES);

    static ObjectPool<object> bt4Pool = new(
      () => 
      {
        lock (creatorObj)
        {
          return NNEvaluatorDef.FromSpecification("~T3_DISTILL_512_15_FP16_TRT", "GPU:0").ToEvaluator();
        }
      }, 
      MAX_BT4_THREADS);

    public static float StockfishSearch(Position pos, SearchLimit limit)
    {
      if (false)
      {
        NNEvaluator evaluator = (NNEvaluator)bt4Pool.GetFromPool();
        NNEvaluatorResult evalResult = evaluator.Evaluate(pos);
        bt4Pool.RestoreToPool(evaluator);
        return evalResult.V;
      }
      else
      {

        GameEngineUCI engine = (GameEngineUCI)sfEnginePool.GetFromPool();
        GameEngineSearchResult searchResult = engine.Search(new PositionWithHistory(pos), limit);
        sfEnginePool.RestoreToPool(engine);
        return searchResult.ScoreQ;
      }
    }

    public static void RewriteBatchQ(IPositionEvaluationBatch batch,
                                     Func<int, Position> getPositionFunc,
                                     float evalTimeStockfish)
    {
//      Console.WriteLine("batch size = " + batch.NumPos);
      PositionEvaluationBatch batchDirect = batch as PositionEvaluationBatch;
      if (batchDirect == null)
      {
        throw new NotImplementedException("RewriteBatchQ only implemented for PositionEvaluationBatch");
      }

      Parallel.For(0, batch.NumPos, new ParallelOptions() { MaxDegreeOfParallelism = MAX_SF_ENGINES }, i =>
      {
        Position pos = getPositionFunc(i);
        if (pos != default)
        {
          float scoreQ = StockfishSearch(pos, new SearchLimit(SearchLimitType.SecondsPerMove, evalTimeStockfish));
          //          Console.Write(batchDirect.GetWin1P(i) + " " + batchDirect.GetLoss1P(i) + " --> ");
          batchDirect.SetWL(i, scoreQ);
        }
        //          Console.WriteLine(sfResult.ScoreQ + " --> " + batchDirect.GetWin1P(i) + " " + batchDirect.GetLoss1P(i));
      });
    }


    static string TB_PATH => CeresUserSettingsManager.Settings.TablebaseDirectory;
    static int SF_HASH_SIZE_MB() => HardwareManager.MemorySize > (256L * 1024 * 1024 * 1024) 
                                                                ? 4_096
                                                                :   512;

    static string SF17_EXE => SoftwareManager.IsLinux ? @"/home/david/apps/SF/sf16.1"
                                                      : @"\\synology\dev\chess\engines\stockfish17-windows-x86-64-avx2.exe";

    static List<string> extraUCI = null;// new string[] {"setoption name Contempt value 5000" };


    public static GameEngineDef MakeEngineDefStockfish(string id,
                                                       string exePath,
                                                       int numThreads,
                                                       bool withTablebase = true,
                                                       int hashtableSize = -1)
    {
      return new GameEngineDefUCI(id, new GameEngineUCISpec(id, exePath, numThreads,
                           hashtableSize == -1 ? SF_HASH_SIZE_MB() : hashtableSize,
                           withTablebase ? TB_PATH : null, uciSetOptionCommands: extraUCI));
    }

    public static GameEngineUCI CreateSFEngine(int numThreads, bool withTablebases)
      => MakeEngineDefStockfish("SF17", SF17_EXE, numThreads, withTablebases, SF_HASH_SIZE_MB()).CreateEngine() as GameEngineUCI;
  }
}

