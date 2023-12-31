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

using static TorchSharp.torch;
using MathNet.Numerics.LinearAlgebra;

#endregion

namespace CeresTrain.NNIntrospection
{
  /// <summary>
  /// Static helper methods for estimating similarity between vectors.
  /// </summary>
  public static class VectorSimilarity
  {
    /// <summary>
    /// Computes array of cosine similarities between specified activations of attention heads
    /// TODO: Is a transpose necessary on attentionOutput here?
    /// </summary>
    /// <param name="H_cat"></param>
    /// <param name="attentionOutput"></param>
    /// <param name="dimModel"></param>
    /// <param name="numHeads"></param>
    /// <param name="dimPerHead"></param>
    /// <returns></returns>
    public static float[] CalcAttentionHeadOutputCosineOutputSimilarities(Tensor H_cat, Tensor attentionOutput, int dimModel, int numHeads, int dimPerHead)
    {
#if NOT
      // ** Pytorch code equivalent, believed working.
// after this line:        H_cat = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
      // NOTE: bias ignored here
      A = H_cat.transpose(2,1).reshape(-1, self.num_heads, self.d_k)
      A_expanded = A.unsqueeze(2)
      B_blocks = self.W_h.weight.transpose(0,1).view(self.num_heads, self.d_k, self.d_model).unsqueeze(0)
      result = torch.matmul(A_expanded, B_blocks)
      result = result.squeeze(2)
      normalized_x = torch.nn.functional.normalize(result, p=2, dim=2)
      cosine_similarity = torch.bmm(normalized_x, normalized_x.transpose(1, 2))

      avg_cosine_similarity = cosine_similarity.mean()
      print (avg_cosine_similarity)

      NOTE: better ending to cut out self-diagonal
        // Create a mask to zero-out the diagonal since we don't want to include self-similarity
        mask = torch.eye(12).to(cos_sim.device).bool()
        // Expand mask to match the batch size and use it to mask the diagonal
        mask = mask.unsqueeze(0).expand(cos_sim.size(0), -1, -1)
        cos_sim.masked_fill_(mask, 0)

        // Compute the mean cosine similarity, excluding the diagonal (self-similarity)
        mean_sim = cos_sim.sum(dim=(1, 2)) / (12 * (12 - 1))

        // mean_sim is now a tensor of shape (2048,) with the average cosine similarity for each of the 2048 elements
        mean_sim.shape, mean_sim
#endif

      using (NewDisposeScope())
      {
        Tensor A = H_cat.transpose(2, 1).reshape(-1, numHeads, dimPerHead);
        Tensor A_expanded = A.unsqueeze(2);
        Tensor B_blocks = attentionOutput.transpose(0, 1).view(numHeads, dimPerHead, dimModel).unsqueeze(0);
        Tensor result = matmul(A_expanded, B_blocks);
        result = result.squeeze(2);

        // MISSING from Torchsharp      Tensor normalized_x = result;// torch.nn.functional.normalize(result, p: 2, dim:2);
        Tensor denom = result.norm(dim: 2, keepdim: true, p: 2).clamp_min_(1E-7).expand_as(result);
        Tensor normalized_x = div(result, denom);

        // TODO: use method above to exclude the diagonal (self-similarity)
        Tensor cosine_similarity = bmm(normalized_x, normalized_x.transpose(1, 2));

        // Create a mask to zero-out the diagonal since we don't want to include self-similarity
        Tensor mask = eye(numHeads, numHeads, ScalarType.Bool).unsqueeze(0).expand(cosine_similarity.size(0), -1, -1);
        cosine_similarity.masked_fill_(mask, 0);

        // Compute the mean cosine similarity, excluding the diagonal (self-similarity)
        Tensor mean_sim = cosine_similarity.sum(new long[] { 1, 2 }) / (numHeads * (numHeads - 1));

        // Convert to float array to return
        float[] hout = mean_sim.cpu().to(ScalarType.Float32).data<float>().ToArray();

        return hout;
      }
    }


    /// <summary>
    /// Based on "Similarity of Neural Network Representations Revisited" by Kornblith et al.
    /// TODO: needs to be matrix based, across many samples.
    /// WARNING: Usefulness of CKA questioned in "Reliability of CKA as a Similarity Measure for Deep Learning" by Davari
    /// </summary>
    /// <param name="vector1"></param>
    /// <param name="vector2"></param>
    /// <returns></returns>
    public static float CenteredKernelAlignment(Vector<float> vector1, Vector<float> vector2)
    {
      int n = vector1.Count;
      Vector<float> ones = Vector<float>.Build.Dense(n, 1f);
      Vector<float> h_vector1 = vector1 - 1f / n * ones * ones.DotProduct(vector1);
      Vector<float> h_vector2 = vector2 - 1f / n * ones * ones.DotProduct(vector2);

      float kernelAlignment = (float)(h_vector1.DotProduct(h_vector2) / (h_vector1.L2Norm() * h_vector2.L2Norm()));
      return kernelAlignment;
    }

  }
}
