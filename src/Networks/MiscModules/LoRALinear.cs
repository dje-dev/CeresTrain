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
using System.Collections.Generic;

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
    public readonly Module<Tensor, Tensor> WrappedLinear;

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

    int in_features;

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
      WrappedLinear = originalLayer;

      int out_features = (int)(originalLayer as Linear).weight.shape[0];
      in_features = (int)(originalLayer as Linear).weight.shape[1];

      const int MinRank = 4;
      Rank = Math.Max(MinRank, in_features / rankDivisor);
      Rank = Math.Min(out_features, Rank); // Ensure rank <= output size

      LoraA = torch.nn.Parameter(torch.empty(in_features, Rank));
      LoraA.name = "lora_A_";
      LoraA.requires_grad = true;

      LoraB = torch.nn.Parameter(torch.empty(Rank, out_features));
      LoraB.name = "lora_B_";
      LoraB.requires_grad = true;

      // Compute the LoRA update and scale it by alpha/sqrt(r)
      // (see "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA" by Kalajdzievski)
      LoraAlpha = Parameter(torch.empty(1));  // Note that we make this trainable, different from most implementations
      LoraAlpha.name = "lora_Alpha_";
      LoraAlpha.requires_grad = true;

      InitializeParameterValues();

      RegisterComponents();
    }

    public static Module<Tensor, Tensor> BaseLinear(Module<Tensor, Tensor> module)
    {
      return (module is Linear) ? module : (module as LoRALinear).WrappedLinear;
    }


    /// <summary>
    /// Overwrites the contents of LoraA, LoraB, LoraAlpha in-place 
    /// with the usual LoRA initialization logic.
    /// </summary>
    public int InitializeParameterValues()
    {
      using (torch.no_grad())
      {
        // Standard deviation for random init
        var std_dev = 1 / torch.sqrt(torch.tensor(Rank));

        // LoraA: random ~ N(0, std_dev^2)
        LoraA.copy_(torch.randn(in_features, Rank).mul_(std_dev));

        // LoraB: zero initialization
        LoraB.zero_();

        // LoraAlpha: alpha = rank / sqrt(rank)
        float alpha = Rank / MathF.Sqrt(Rank);
        LoraAlpha.fill_(alpha);

        return 3;
      }
    }

    public override Tensor forward(Tensor x)
    {
      if (LoRAEnabledFunc == null || LoRAEnabledFunc())
      {
        using (NewDisposeScope())
        {
          // Compute LoRA update
          Tensor loraUpdate = LoraA.matmul(LoraB);

          // Apply LoRA
          Tensor original = WrappedLinear.forward(x);
          Tensor ret = original + LoraAlpha * (x.matmul(LoraA).matmul(LoraB));
          return ret.MoveToOuterDisposeScope();
        }
      }
      else
      {
        return WrappedLinear.forward(x);
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
    /// Reinitializes the LoRA weights (A, B, and alpha) for all LoRALinear modules in a given network. 
    /// </summary>
    /// <param name="netModule"></param>
    /// <returns></returns>
    public static int ReinitializeLoRAWeights(Module netModule)
    {
      int numReinitialized = 0;
      foreach (Module module in netModule.modules())
      {
        if (module is LoRALinear)
        {
          numReinitialized += (module as LoRALinear).InitializeParameterValues();
        }
      }

      return numReinitialized;
    }


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
          if (loraLinear.WrappedLinear is Linear originalLinear)
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
    /// Extracts Dictionary of LoRA parameters from a model.
    /// The LoRA wrapped layers are identified by the presence of the string "LoraAlpha" in their name.
    /// </summary>
    /// <param name="model"></param>
    /// <returns></returns>
    public static Dictionary<string, (float, float[,], float[,])> ExtractedLoRAParameters(Module model)
    {
      Dictionary<string, (float, float[,], float[,])> loraParams = new();
      foreach ((string name, Parameter parameter) np in model.named_parameters(true))
      {
        if (np.name.Contains("LoraAlpha"))
        {
          string onnxName = np.name.Replace(".LoraAlpha", "").Replace(".LoraA", "").Replace(".LoraB", "");

          onnxName = onnxName.Replace("encoder_layer_", "/transformer_layer.");
          onnxName = onnxName.Replace("q2", "/attention/q2/MatMul");
          onnxName = onnxName.Replace("k2", "/attention/k2/MatMul");
          onnxName = onnxName.Replace("mlpLinear1", "/mlp/linear1/MatMul");
          onnxName = onnxName.Replace("mlpLinear2", "/mlp/linear2/MatMul");
          onnxName = onnxName.Replace("./", "/");
          //Console.WriteLine(onnxName + "   " + np.name);

          Parameter pA = model.get_parameter(np.name.Replace("LoraAlpha", "LoraA"));
          Parameter pB = model.get_parameter(np.name.Replace("LoraAlpha", "LoraB"));
          Parameter pAlpha = np.parameter;

          float alpha = pAlpha.to(ScalarType.Float32).data<float>()[0];
          float[] a = pA.to(ScalarType.Float32).data<float>().ToArray();
          float[] b = pB.to(ScalarType.Float32).data<float>().ToArray();
          float[,] a2D = ArrayUtils.To2D(a, (int)pA.shape[1]);
          float[,] b2D = ArrayUtils.To2D(b, (int)pB.shape[1]);

          loraParams[onnxName] = (alpha, a2D, b2D);
        }
      }

      return loraParams;
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
