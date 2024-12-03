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
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.MTCSNodes;

using CeresTrain.TPG;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.MCTS.LeafExpansion;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  public static class TPGExtractorFromTree
  {
    /// <summary>
    /// Extracts List of TPGRecord from tree having a minimum specified number of visits.
    /// </summary>
    /// <param name="minNodesRequired"></param>
    /// <param name="fillInTPGReader"></param>
    /// <param name="numNonFillInTPGs"></param>
    /// <param name="acceptNodePredicate"></param>
    /// <returns></returns>
    public static List<TPGRecord> ExtractTPGsFromTree(MCTSTree tree,
                                                      int minNodesRequired,
                                                      TPGFileReader fillInTPGReader,
                                                      out int numNonFillInTPGs,
                                                      Predicate<MCTSNode> acceptNodePredicate = null)
    {
      List<TPGRecord> ret = new();

      int count = 0;
      int countNonFillInTPGs = 0;

      MCTSNode rootNode = tree.Root;
      rootNode.Context.Root.StructRef.Traverse(rootNode.Context.Tree.Store,
        (ref MCTSNodeStruct nodeRef, int depth) =>
        {
          //     if (nodeRef.N >= minN && FP16.IsNaN(nodeRef.VSecondary))// && !nodeRef.IsTranspositionLinked)
          if (nodeRef.N < minNodesRequired)
          {
            // We must be done, children will all have N strictly less than minNodes.
            return false;
          }
          else
          {
            MCTSTree tree = rootNode.Context.Tree;
            MCTSNode node = tree.GetNode(nodeRef.Index);

            if (acceptNodePredicate == null || acceptNodePredicate(node))
            {
              float diff = (float)Math.Abs(rootNode.Q - nodeRef.Q * (depth % 2 == 1 ? -1 : 1));
              // Console.WriteLine((diff > 0.25 ? "*" : " ") + LastSearchResult.ScoreQ + " --> " + nodeRef.N + " " + depth + " " + nodeRef.Q);

              const bool EMIT_LAST_PLY_SINCE_SQUARE = false;
              TPGRecord tpgRecord = TPGExtractorFromNode.ExtractTPGRecordFromNode(tree, node, EMIT_LAST_PLY_SINCE_SQUARE);
              ret.Add(tpgRecord);
              countNonFillInTPGs++;

              if (fillInTPGReader != null)
              {
                lock (fillInTPGReader)
                {
                  // It is assumed that the tpgReader is configured to 
                  // return batches of the appropriate size for fill-in.
                  TPGRecord[] fillInTPG = fillInTPGReader.NextBatch();
                  for (int i = 0; i < fillInTPG.Length; i++)
                  {
                    ret.Add(fillInTPG[i]);
                  }
                }
              }

              count++;
            }
            return true;
          }
        }, Ceres.Base.DataType.Trees.TreeTraversalType.DepthFirst);

      numNonFillInTPGs = countNonFillInTPGs;
      return ret;
    }
  }
}
