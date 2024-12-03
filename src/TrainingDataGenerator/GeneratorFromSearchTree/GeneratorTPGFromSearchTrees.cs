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
using System.Linq;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.Threading;

using Zstandard.Net;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.Positions;
using Ceres.Chess.EncodedPositions;
using CeresTrain.TPG;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using CeresTrain.TrainingDataGenerator.CeresTrain.TrainingDataGenerator;
using Ceres.MCTS.MTCSNodes;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  public static class GeneratorTPGFromSearchTrees
  {
    public delegate bool RunSearchFromTrainingPositionPredicate(in EncodedTrainingPosition trainingPos);

    /// <summary>
    /// Generates TPG files by:
    ///   - scanning LC0 training data and selecting a subset of positions
    ///   - running a search from each position using a specified NNEvaluator
    ///   - traversing tree looking for nodes with sufficiently large number of visits (N)
    ///   - writing these nodes to a TPG file along with derived training targets (value and policy)
    /// </summary>
    /// <param name="trainingDataEvaluators"></param>
    /// <param name="parms"></param>
    /// <exception cref="Exception"></exception>
    public static void GenerateTPGsFromSearchTrees(PositionEvalAccuracyEstimatorBySearch[] trainingDataEvaluators,
                                                   GeneratorTPGFromSearchTreesParams parms)
    {
      TPGFileReader tpgReader = null;
      if (parms.FillInNumTPGPerGeneratedTPG > 0)
      {
        string[] tpgFileNames = Directory.GetFiles(parms.FillInTPGsDirectoryName, "*.zst");
        if (tpgFileNames.Length == 0)
        {
          throw new Exception($"No TPG files (with extension ZST) found in specified directory {parms.FillInTPGsDirectoryName}");
        }

        // Pick a file at random.
        // Note that the TPG files already contain shuffled positions, so we don't need to shuffle them here.
        // TODO: currently we only support reading from a single file and fail if the file is exhausted. Improve.
        string tpgFileName = tpgFileNames[Random.Shared.Next(tpgFileNames.Length)];
        Console.WriteLine($"Sourcing fill-in TPG data from {tpgFileName}");
        tpgReader = new TPGFileReader(tpgFileName, parms.FillInNumTPGPerGeneratedTPG);
      }

      Console.WriteLine($"EvaluateTARS {parms.SourceTARsDirectoryName} num positions {parms.TargetNumNonFillInTPGToGenerate} using {trainingDataEvaluators[0].EvaluatorDef}");

      string[] allFiles = Directory.GetFiles(parms.SourceTARsDirectoryName, "*.tar").ToArray();
      if (allFiles.Length == 0)
      {
        throw new Exception($"No files *.tar files found in specified directory {parms.SourceTARsDirectoryName}");
      }
      ObjUtils.Shuffle(allFiles);

      int numFilesProcessed = 0;
      int numTPGEmitted = 0;

      float sumValueErrors = 0;
      float sumPolicyErrors = 0;

      // Create compressed output stream for TPGs.
      FileStream fileStream = new(parms.OutputTPGName, FileMode.Create);
      ZstandardStream outputStream = new(fileStream, parms.CompressionLevel);

      foreach (string tarFileName in allFiles)
      {
        // Skip file if specified filter disallows it.
        if (parms.PredicateAcceptFile != null && !parms.PredicateAcceptFile(tarFileName))
        {
          break;
        }

        // Exit loop if we have collected as many positions as requested.
        int numTPGRemainingToGenerate = parms.TargetNumNonFillInTPGToGenerate - numTPGEmitted;
        if (numTPGRemainingToGenerate <= 0)
        {
          break;
        }

        numTPGEmitted += GenerateTPGFromTAR(tarFileName, outputStream, trainingDataEvaluators,
                                            parms.TARPosSkipCount, numTPGRemainingToGenerate,
          delegate (in EncodedTrainingPosition trainingPos)
          {
            EncodedPositionEvalMiscInfoV6 trainingInfo = trainingPos.PositionWithBoards.MiscInfo.InfoTraining;
            return trainingInfo.Uncertainty >= parms.ThresholdUncertaintyRunSearch;
          },
          parms.NodesPerSearch, parms.SearchLimitSF, parms.MinNodesWriteAsTPG, tpgReader);
      }

      outputStream.Close();
      fileStream.Close();
      tpgReader?.Shutdown();
    }



    static int GenerateTPGFromTAR(string sourceTARFileName, 
                                  Stream outputStream,
                                  IEnumerable<PositionEvalAccuracyEstimatorBySearch> trainingDataEvaluators,
                                  int skipCount, 
                                  int maxTPGToGenerate,
                                  RunSearchFromTrainingPositionPredicate runSearchFromTrainingPositionDelegate,
                                  int numSearchNodes, 
                                  SearchLimit searchLimitSF, 
                                  int minNumNodesWriteAsTPG,
                                  TPGFileReader tpgReader)
    {
      int numSeen = 0;
      int numUsed = 0;
      int numNonFillInTPGs = 0;

      int numEvaluators = trainingDataEvaluators.Count();

      // Build concurrency-safe collection of all evaluators.
      BlockingCollection<PositionEvalAccuracyEstimatorBySearch> evaluators = new();
      trainingDataEvaluators.ToList().ForEach(eval => evaluators.Add(eval));

      foreach (var position in EncodedTrainingPositionReaderTAREngine.EnumeratedPositions(sourceTARFileName, skipCount: skipCount))
      {
        numSeen++;

        ref readonly EncodedTrainingPosition refPos = ref position.gamePositions.Span[position.indexInGame];

        // Possibly call delegate to determine if this position should be used for search.
        if (!runSearchFromTrainingPositionDelegate(in refPos))
        {
          continue;
        }
        else
        {
          numUsed++;
          Task task = LaunchTaskToStartSearchAndTPGGeneration(numSearchNodes, searchLimitSF,
                                                              taskNumNonFillInTPGS => Interlocked.Add(ref numNonFillInTPGs, taskNumNonFillInTPGS),
                                                              evaluators, refPos, (lastSearch) => true, minNumNodesWriteAsTPG,
                                                              tpgReader, outputStream);
          //pwh.Dump();
        }

        if (numNonFillInTPGs >= maxTPGToGenerate)
        {
          break;
        }
      }

      // Wait until all the evaluators are put back into queue, indicating no searches in process.
      while (evaluators.Count() < trainingDataEvaluators.Count())
      {
        Thread.Sleep(30);
      }

      Console.WriteLine();

      return numNonFillInTPGs;
    }



    static Task LaunchTaskToStartSearchAndTPGGeneration(int numSearchNodes, SearchLimit searchLimitSF,
                                                        Action<int> registerNumGeneratedNonFillInTPGs,
                                                        BlockingCollection<PositionEvalAccuracyEstimatorBySearch> evaluators,
                                                        EncodedTrainingPosition refPos,
                                                        Predicate<PositionEvalAccuracyEstimatorBySearch> lastSearchResultShouldBeWrittenAsTPG,
                                                        int minNumNodesForNodeToBeWrittenAsTPG,
                                                        TPGFileReader fillInTPGReader,
                                                        Stream writeTPGStream)
    {
      PositionEvalAccuracyEstimatorBySearch thisEvaluator;
      thisEvaluator = evaluators.Take();

      Task task = new(() =>
      {
        try
        {
          // Get start position for search.
          PositionWithHistory pwh = refPos.ToPositionWithHistory(8);

          // Run search on this position.
          bool searchCompleted = thisEvaluator.DoSearchEvaluation(pwh, SearchLimit.NodesPerMove(numSearchNodes), searchLimitSF, in refPos);
          bool writeTPG = searchCompleted && (lastSearchResultShouldBeWrittenAsTPG == null || lastSearchResultShouldBeWrittenAsTPG(thisEvaluator));
          if (searchCompleted && (lastSearchResultShouldBeWrittenAsTPG == null || lastSearchResultShouldBeWrittenAsTPG(thisEvaluator)))
          {
            //trainingDataEvaluator.LastSearchResult.Search.Manager.DumpFullInfo(Console.Out, "UCI");

            // Extract set of TPGRecords taken from the search tree.
            MCTSNode rootNode = thisEvaluator.LastSearchResult.Search.SearchRootNode;
            List<TPGRecord> tpgsNew = TPGExtractorFromTree.ExtractTPGsFromTree(rootNode.Tree, minNumNodesForNodeToBeWrittenAsTPG, fillInTPGReader, out int numNonFillInTPGs);

            // Write TPGs to output stream.
            ReadOnlySpan<byte> bufferAsBytes = MemoryMarshal.Cast<TPGRecord, byte>(tpgsNew.ToArray().AsSpan());
            lock (writeTPGStream)
            {
              writeTPGStream.Write(bufferAsBytes);
            }

            // Notify caller of number of TPGs generated.
            registerNumGeneratedNonFillInTPGs(numNonFillInTPGs);
            float yieldFactorAboveSimpleDivision = tpgsNew.Count / ((float)thisEvaluator.LastSearchResult.FinalN / minNumNodesForNodeToBeWrittenAsTPG);
            Console.WriteLine("TPGS: " + thisEvaluator.LastSearchResult.ScoreQ + " -->  yield factor= " + yieldFactorAboveSimpleDivision
              + " Search " + thisEvaluator.LastSearchResult.Search.SearchRootNode.N + " find " + numNonFillInTPGs + " nodes of N >= " + minNumNodesForNodeToBeWrittenAsTPG
              + " with total numTPG (including fill-in) of " + tpgsNew.Count);
          }

          //        WriteTPGs(numTPGS, tpgBuffer);

          // Put back the evaluator
          evaluators.Add(thisEvaluator);
        }
        catch (Exception exc)
        {
          Console.WriteLine("Exception in LaunchTaskToStartSearchAndTPGGeneration");
          Console.WriteLine(exc);
          Environment.Exit(3);
        }
      });

      task.Start();
      return task;
    }

#if NOT
    static void WriteTPGs(int numTPGS, TPGRecord[] tpgBuffer)
    {
      Console.WriteLine("writing ");
      using (new TimingBlock("write/compress"))
      {
        string fn_suffix = (Environment.TickCount % 1000).ToString();
        TPGRecord.WriteToZSTFile(@"f:\new_tpg_" + fn_suffix + ".zst", tpgBuffer, numTPGS);
      }
    }
#endif

  }
}
