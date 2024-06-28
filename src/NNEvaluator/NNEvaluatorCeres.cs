#region Using directives

using System;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace CeresTrain.NNEvaluators
{
  public class NNEvaluatorCeres : NNEvaluator
  {
    public override bool IsWDL => true;
    public override bool HasM => true;
    public override bool HasAction => true;
    public override bool HasUncertaintyV => true;
    public override bool HasUncertaintyP => true;
    public override bool HasValueSecondary => true;
    public override int MaxBatchSize => 1024;

    public NNEvaluatorCeres(string fn, object options)
    {

    }


    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      throw new NotImplementedException();
    }

    protected override void DoShutdown()
    {
      throw new NotImplementedException();
    }
  }

}
