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
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

using Ceres.Base.DataType;

#endregion


namespace CeresTrain.Networks.MiscModules
{
  /// <summary>
  /// A PyTorch module for applying Low-Rank Adaptation (LoRA) to a linear layer.
  ///   
  /// LoRA introduces trainable low-rank matrices (A and B) to fine-tune large
  /// pre-trained models efficiently. This reduces the number of trainable 
  /// parameters. This implementation replaces a standard linear layer 
  /// with a LoRA-enabled linear layer.

  /// Parameters:
  /// - original_layer: The original nn.Linear layer to be adapted with LoRA.
  /// - rank_divisor: Determines the rank of the low-rank decomposition. The rank is 
  ///   calculated as the in_features divided by the rank_divisor.
  /// - enable_lora: If True, enables LoRA by adding the low-rank matrices A and B.
  /// 
  /// Reference:
  /// "LoRA: Low-Rank Adaptation of Large Language Models" by Hu et. al. (2021)
  /// https://arxiv.org/abs/2106.09685
  /// </summary>
  public class LoRALinear : Module<Tensor, Tensor>
  {
    public Func<bool> LoRAEnabledFunc;

    /// <summary>
    /// The original layer to be adapted with LoRA.
    /// </summary>
    public readonly Module<Tensor, Tensor> WrapppedLinear;

    /// <summary>
    /// Rank of the low-rank decomposition.
    /// </summary>
    public readonly int Rank;


    // LoRA trainable parameters      
    // Note that the specific names "lora_A", "lora_B" and "lora_alpha" are
    // referenced elsewhere

    /// <summary>
    /// Scaling factor for the LoRA update.
    /// </summary>
    public readonly Parameter LoraAlpha;

    /// <summary>
    /// Low-rank matrix A.
    /// </summary>
    public readonly Parameter LoraA;

    /// <summary>
    /// Low-rank matrix B.
    /// </summary>
    public readonly Parameter LoraB;


    static int counter = 0;



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="originalLayer"></param>
    /// <param name="rankDivisor"></param>
    /// <param name="loRAEnabledFunc"></param>
    public LoRALinear(Module<Tensor, Tensor> originalLayer, int rankDivisor, Func<bool> loRAEnabledFunc = null)
      : base("LoRALinear")
    {
      LoRAEnabledFunc = loRAEnabledFunc;
      WrapppedLinear = originalLayer;

      int out_features = (int)(originalLayer as Linear).weight.shape[0];
      int in_features = (int)(originalLayer as Linear).weight.shape[1];

      const int MinRank = 4;
      Rank = Math.Max(MinRank, in_features / rankDivisor);
      Rank = Math.Min(out_features, Rank); // Ensure rank <= output size

      Tensor std_dev = 1 / torch.sqrt(torch.tensor(Rank));

      LoraA = torch.nn.Parameter(torch.randn(in_features, Rank) * std_dev);
      LoraA.name = "lora_A_";
      LoraA.requires_grad = true;

      LoraB = torch.nn.Parameter(torch.zeros(Rank, out_features));
      LoraB.name = "lora_B_";
      LoraB.requires_grad = true;

      // Compute the LoRA update and scale it by alpha/sqrt(r)
      // (see "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA" by Kalajdzievski)
      float alpha = (Rank / MathF.Sqrt(Rank));
      LoraAlpha = Parameter(tensor(alpha));  // Note that we make this trainable, different from most implementations
      LoraAlpha.name = "lora_Alpha_";
      LoraAlpha.requires_grad = true;

      RegisterComponents();
    }


    public override Tensor forward(Tensor x)
    {
      if (LoRAEnabledFunc == null || LoRAEnabledFunc())
      {
        // Compute LoRA update
        Tensor loraUpdate = LoraA.matmul(LoraB);

        // Apply LoRA
        Tensor original = WrapppedLinear.forward(x);
        return original + LoraAlpha * (x.matmul(LoraA).matmul(LoraB));
      }
      else
      {
        return WrapppedLinear.forward(x);
      }
    }


    #region Helper methods

    /// <summary>
    /// Possibly wrap a module in a LoRALinear layer.
    /// 
    /// N.B. currently limited to Linear layers.
    /// </summary>
    /// <param name="module"></param>
    /// <param name="lora_rank_divisor"></param>
    /// <returns></returns>
    public static Module<Tensor, Tensor> PossiblyLoRAWrappedModule(Module<Tensor, Tensor> module, 
                                                                   int lora_rank_divisor,
                                                                   Func<bool> loraEnabledFunc = null)
      => lora_rank_divisor > 0 ? new LoRALinear(module, lora_rank_divisor, loraEnabledFunc) 
                               : module; 


    /// <summary>
    /// Sets the weights and biases for a given layer, handling both Linear and LoRALinear types.
    /// </summary>
    /// <param name="layer">The layer to set weights and biases for.</param>
    /// <param name="weights">The weight array.</param>
    /// <param name="bias">The bias array.</param>
    /// <param name="device">The device to place the tensor on.</param>
    /// <param name="dtype">The data type of the tensor.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the layer type is unsupported or if LoRALinear's OriginalLayer is not Linear.
    /// </exception>
    public static void SetLayerWeightsAndBiases(Module layer,
                                                float[,] weights, float[] bias,
                                                Device device, ScalarType dtype)
    {
      using (torch.no_grad())
      {
        if (layer is Linear linear)
        {
          linear.weight.set_(torch.tensor(weights, dtype: dtype, device: device));
          linear.bias.set_(torch.tensor(bias, dtype: dtype, device: device));
        }
        else if (layer is LoRALinear loraLinear)
        {
          if (loraLinear.WrapppedLinear is Linear originalLinear)
          {
            originalLinear.weight.set_(torch.tensor(weights, dtype: dtype, device: device));
            originalLinear.bias.set_(torch.tensor(bias, dtype: dtype, device: device));
          }
          else
          {
            throw new InvalidOperationException("LoRALinear's OriginalLayer is not of type Linear.");
          }
        }
        else
        {
          throw new InvalidOperationException($"Unsupported layer type: {layer.GetType().Name}");
        }
      }
    }



    /// <summary>
    /// Returns the LoRA weights (alpha and A, B) matrices for the layer.
    /// </summary>
    /// <returns></returns>
    public (float alpha, float[,] A, float[,] B) GetLoRAWeights()
    {
      float alpha = LoraAlpha.cpu().to(ScalarType.Float32).data<float>().ToArray()[0];
      float[,] aMatrix = ArrayUtils.To2D<float>(LoraA.cpu().to(ScalarType.Float32).data<float>().ToArray(), (int)LoraA.shape[1]);
      float[,] bMatrix = ArrayUtils.To2D<float>(LoraB.cpu().to(ScalarType.Float32).data<float>().ToArray(), (int)LoraB.shape[1]);
      return (alpha, aMatrix, bMatrix);
    }


    #endregion
  }

}
