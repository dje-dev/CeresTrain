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

using TorchSharp;
using static TorchSharp.torch;

#endregion

namespace CeresTrain.NNIntrospection
{
  /// <summary>
  /// Static helper methods for calculating the intrinsic dimensionality of a dataset.
  /// </summary>
  public static class IntrinsicDimensionality
  {
    /// <summary>
    /// Calculates the intrinsic dimension of the the input, using TwoNN algorithm from:
    ///   [1] E. Facco, M. d’Errico, A. Rodriguez & A. Laio
    ///   Estimating the intrinsic dimension of datasets by a minimal
    ///   neighborhood information(https://doi.or/g/10.1038/s41598-017-11873-y).
    ///   
    /// Based on a Python implementation from:
    ///   https://github.com/VRehnberg/torch-twonn/blob/main/twonn.py
    /// </summary>
    /// <param name="points">Size is (batch_size ×) n_points × embedding_dimension</param>
    /// <param name="fit_fraction">Fraction of points to use in fit (more reliable to use value slightly less than 1)</param>
    /// <returns></returns>
    public static float[] TwoNN(Tensor points, float fit_fraction = 0.9f)
    {
      if (points.shape.Length > 3 || points.shape.Length < 2)
      {
        throw new ArgumentException("Input should be 2 or 3 dimensional.");
      }

      // Massage points tensor
      points = points.to(torch.float32);
      if (points.shape.Length == 2)
      {
        points = points.unsqueeze(0);
      }

      // Get information from points
      long batch_size = points.shape[0];
      long n_points = points.shape[^2];
      long n_dim = points.shape[^1];

      if (n_points < 3)
      {
        throw new ArgumentException("TwoNN needs atleast three points to work.");
      }

      if (fit_fraction > 1.0 || fit_fraction <= 0.0)
      {
        throw new ArgumentException("Parameter fit_fraction must be in (0, 1].");
      }

      using (NewDisposeScope())
      {
        // Compute pairwise distances.
        // N.B. For reasons of numerical stability, must specify compute mode explicitly to not use matrix multiplication.
        //      See: https://github.com/pytorch/pytorch/issues/42479
        var distances = torch.cdist(points, points, p: 2, compute_mode: compute_mode.donot_use_mm_for_euclid_dist);
        (distances, _) = distances.topk(3, dim: -1, largest: false);

        // Compute µ = r_2 / r_1
        Tensor[] r = torch.split(distances, new long[] { 1, 1, 1 }, dim: -1);
        (Tensor r0, Tensor r1, Tensor r2) = (r[0], r[1], r[2]);
        Tensor mu = r2 / r1;

        //    if (mu <= 1.0f)
        //    {
        //      Tensor allClose = torch.isclose(mu, torch.ones(1, device: device)))
        //      throw new Exception("Something went wrong when computing µ.");
        //    }
        //    if (!((mu > 1.0f || (torch.isclose(mu, torch.ones(1, device: device))))).all()
        //        throw new Exception("Something went wrong when computing µ.");

        // Compute the empirical cumulate
        Tensor empirical = (torch.arange(n_points) / n_points).tile(new long[] { batch_size, 1 }).unsqueeze(2);
        empirical = empirical.to(mu.dtype).to(mu.device);
        (mu, _) = mu.sort(dim: 1);

        // Fit the the intrinsic dimension
        // d = - log(1 - F(µ)) / log(µ)
        Tensor y_full = -torch.log(1.0 - empirical);
        Tensor x_full = torch.log(mu);


        int n_fit = (int)(round(fit_fraction * n_points).cpu().item<float>());
        Tensor y_fit = torch.narrow(y_full, 1, 0, n_fit);
        Tensor x_fit = torch.narrow(x_full, 1, 0, n_fit);

        //    Tensor y_fit = y_full.slice( y_full[:, :n_fit];
        //    Tensor x_fit = x_full[:, :n_fit];

        // Here assume that the values of log(1 - F(µ)) are exact and
        // log(µ) is drawn from a normal distribution (prob. not correct).
        // I.e. 1 / d* = argmin_(1 / d) ||(-log(1 - F(µ))) (1 / d) - µ||_2 )
        Tensor inv_d = torch.bmm(torch.pinverse(y_fit), x_fit);
        Tensor intrinsic_dimension = 1.0 / torch.narrow(inv_d, 1, 0, 1).squeeze().cpu(); //    intrinsic_dimension = 1.0 / inv_d[:, 0]

        return points.shape.Length == 2 ? new float[] { intrinsic_dimension.item<float>() } : intrinsic_dimension.data<float>().ToArray();
      }
    }
  }

}
