#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/cerestrain.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with CeresTrain. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion


// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;

namespace CeresTrain.Optimizers
{
  public static class MuonOptimizerHelper
  {
    /// <summary>
    /// Constructs a new MuonOptimizer.
    /// </summary>
    /// <param name="muonParams">Parameters to optimize with the Muon update (must be 2D).</param>
    /// <param name="adamwParams">Parameters to optimize with the AdamW backup update.</param>
    /// <param name="lr">Base learning rate.</param>
    /// <param name="wd">Weight decay.</param>
    /// <param name="momentum">Momentum for the internal SGD.</param>
    /// <param name="nesterov">Whether to use Nesterov momentum.</param>
    /// <param name="nsSteps">Number of Newton–Schulz iterations.</param>
    /// <param name="adamwBeta1">Beta1 for the AdamW backup update.</param>
    /// <param name="adamwBeta2">Beta2 for the AdamW backup update.</param>
    /// <param name="adamwEps">Epsilon for the AdamW backup update.</param>
    /// <returns>A new MuonOptimizer instance.</returns>
    public static MuonOptimizer MuonOptimizer(IEnumerable<Parameter> muonParams, IEnumerable<Parameter> adamwParams = null,  
                                              double lr = 1e-3, double wd = 0.1, double momentum = 0.95, bool nesterov = true, int nsSteps = 5,
                                              double adamwBeta1 = 0.95, double adamwBeta2 = 0.95, double adamwEps = 1e-8)
    {
      return new MuonOptimizer(muonParams, adamwParams, lr, wd, momentum, nesterov, nsSteps, adamwBeta1, adamwBeta2, adamwEps);
    }
  }



  /// <summary>
  /// Implements the Muon optimizer:
  /// 
  /// Internally, for each 2D parameter (typically weights of linear layers) the optimizer first
  /// runs a standard SGD momentum update and then replaces the raw update with an orthogonalized version
  /// computed by a Newton–Schulz iteration. Parameters not meeting the 2D (or “Muon‐eligible”) criteria
  /// are updated using an AdamW backup.
  /// </summary>
  public class MuonOptimizer : OptimizerHelper
  {
    private Options _defaults;
    private List<ParamGroup> _parameter_groups;

    /// <summary>
    /// Constructs a MuonOptimizer from separate parameter lists.
    /// </summary>
    /// <param name="muonParams">Parameters to be updated via Muon logic (2D only).</param>
    /// <param name="adamwParams">Parameters to be updated via AdamW backup.</param>
    /// <param name="lr">Base learning rate.</param>
    /// <param name="wd">Weight decay.</param>
    /// <param name="momentum">Momentum for the internal SGD.</param>
    /// <param name="nesterov">Whether to use Nesterov momentum.</param>
    /// <param name="nsSteps">Number of Newton–Schulz iterations.</param>
    /// <param name="adamwBeta1">Beta1 for AdamW backup.</param>
    /// <param name="adamwBeta2">Beta2 for AdamW backup.</param>
    /// <param name="adamwEps">Epsilon for AdamW backup.</param>
    public MuonOptimizer(IEnumerable<Parameter> muonParams, IEnumerable<Parameter> adamwParams, double lr = 1e-3, double wd = 0.1,
                         double momentum = 0.95, bool nesterov = true, int nsSteps = 5,
                         double adamwBeta1 = 0.95, double adamwBeta2 = 0.95, double adamwEps = 1e-8)
        : this(new List<ParamGroup> { new ParamGroup(muonParams.Union(adamwParams), lr, wd, momentum, nesterov, nsSteps, adamwBeta1, adamwBeta2, adamwEps) })
    {
      List<Parameter> allParams = new List<Parameter>(muonParams);
      if (adamwParams != null)
      {
        allParams.AddRange(adamwParams);
      }

      // For each parameter, set its state flag indicating which update to use.
      foreach (Parameter p in allParams)
      {
        State state = (State)_state[p.Handle];
        if (muonParams.Contains(p))
        {
          state.UseMuon = true;
        }
        else
        {
          state.UseMuon = false;
        }
      }
    }

    /// <summary>
    /// Constructs a MuonOptimizer from parameter groups.
    /// </summary>
    /// <param name="parameters">Parameter groups to optimize.</param>
    /// <param name="lr">Base learning rate.</param>
    /// <param name="wd">Weight decay.</param>
    /// <param name="momentum">Momentum for the internal SGD.</param>
    /// <param name="nesterov">Whether to use Nesterov momentum.</param>
    /// <param name="nsSteps">Number of Newton–Schulz iterations.</param>
    /// <param name="adamwBeta1">Beta1 for AdamW backup.</param>
    /// <param name="adamwBeta2">Beta2 for AdamW backup.</param>
    /// <param name="adamwEps">Epsilon for AdamW backup.</param>
    public MuonOptimizer(IEnumerable<ParamGroup> parameters, double lr = 1e-3, double wd = 0.1,
                         double momentum = 0.95, bool nesterov = true, int nsSteps = 5,
                         double adamwBeta1 = 0.95, double adamwBeta2 = 0.95, double adamwEps = 1e-8)
    {
      if (lr < 0)
      {
        throw new ArgumentException($"Invalid learning rate: {lr}");
      }
      if (wd < 0)
      {
        throw new ArgumentException($"Invalid weight decay: {wd}");
      }
      if (momentum < 0 || momentum > 1)
      {
        throw new ArgumentException($"Invalid momentum: {momentum}");
      }

      _defaults = new Options
      {
        LearningRate = lr,
        wd = wd,
        momentum = momentum,
        nesterov = nesterov,
        nsSteps = nsSteps,
        adamw_beta1 = adamwBeta1,
        adamw_beta2 = adamwBeta2,
        adamw_eps = adamwEps,
        InitialLearningRate = lr
      };

      _parameter_groups = new List<ParamGroup>();

      foreach (ParamGroup group in parameters)
      {
        this.add_param_group(group);
      }
    }

    /// <summary>
    /// Performs a single optimization step.
    /// </summary>
    /// <param name="closure">
    /// A closure that reevaluates the model and returns the loss.
    /// </param>
    /// <returns>The loss (if provided by the closure).</returns>
    public override Tensor step(Func<Tensor> closure = null)
    {
      Tensor loss = null;
      if (closure != null)
      {
        using (IDisposable enabled = torch.enable_grad())
        {
          loss = closure();
        }
      }

      foreach (ParamGroup group in _parameter_groups)
      {
        Options options = group.Options as Options;
        double lr = options.LearningRate.Value;
        double wd = options.wd.Value;
        double momentum = options.momentum.Value;
        bool nesterov = options.nesterov.Value;
        int nsSteps = options.nsSteps.Value;
        double adamw_beta1 = options.adamw_beta1.Value;
        double adamw_beta2 = options.adamw_beta2.Value;
        double adamw_eps = options.adamw_eps.Value;

        // Separate parameters based on whether they use Muon update.
        List<Parameter> muonParams = new List<Parameter>();
        List<Parameter> adamwParams = new List<Parameter>();

        foreach (Parameter p in group.Parameters)
        {
          State state = (State)_state[p.Handle];
          if (state.UseMuon)
          {
            muonParams.Add(p);
          }
          else
          {
            adamwParams.Add(p);
          }
        }

        // --- Muon update ---
        foreach (Parameter p in muonParams)
        {
          Tensor grad = p.grad;
          if ((object)grad == null)
          {
            continue;
          }
          // If gradient has more than 2 dimensions, reshape it to 2D.
          if (grad.ndim > 2)
          {
            long firstDim = grad.size(0);
            grad = grad.view(new long[] { firstDim, grad.numel() / firstDim });
          }

          State state = (State)_state[p.Handle];
          if ((object)state.MomentumBuffer == null)
          {
            state.MomentumBuffer = torch.zeros_like(grad);
          }

          // Update momentum buffer: momentum_buffer = momentum * momentum_buffer + grad
          state.MomentumBuffer.mul_(momentum);
          state.MomentumBuffer.add_(grad);

          Tensor update = null;
          if (nesterov)
          {
            // Nesterov update: g + momentum * momentum_buffer
            update = grad.add(state.MomentumBuffer, momentum);
          }
          else
          {
            update = state.MomentumBuffer;
          }

          // Orthogonalize the update using the Newton–Schulz iteration.
          Tensor u = ZeropowerViaNewtonSchulz5(update, nsSteps);

          // Adjust the learning rate based on parameter shape.
          double adjusted_lr = AdjustLearningRate(lr, p.shape);

          // Apply weight decay.
          p.mul_(1 - lr * wd);

          // Update parameters.
          p.add_(u, -adjusted_lr);
        }

        // --- AdamW backup update ---
        foreach (Parameter p in adamwParams)
        {
          Tensor grad = p.grad;
          if ((object)grad == null)
          {
            continue;
          }
          State state = (State)_state[p.Handle];
          if (state.Step == 0)
          {
            state.AdamwMoment1 = torch.zeros_like(grad);
            state.AdamwMoment2 = torch.zeros_like(grad);
          }
          state.Step += 1;
          long step = state.Step;

          // Update first moment: m = beta1 * m + (1 - beta1) * grad
          state.AdamwMoment1.mul_(adamw_beta1);
          state.AdamwMoment1.add_(grad, 1 - adamw_beta1);

          // Update second moment: v = beta2 * v + (1 - beta2) * (grad^2)
          Tensor gradSquared = grad.mul(grad);
          state.AdamwMoment2.mul_(adamw_beta2);
          state.AdamwMoment2.add_(gradSquared, 1 - adamw_beta2);

          double bias_correction1 = 1 - Math.Pow(adamw_beta1, step);
          double bias_correction2 = 1 - Math.Pow(adamw_beta2, step);
          Tensor denom = state.AdamwMoment2.sqrt().add(adamw_eps);
          Tensor update = state.AdamwMoment1.div(denom);
          double scale = bias_correction1 / Math.Sqrt(bias_correction2);

          p.mul_(1 - lr * wd);
          p.add_(update, -lr / scale);
        }
      }

      return loss;
    }

    /// <summary>
    /// Adjusts the learning rate based on the parameter matrix dimensions.
    /// </summary>
    /// <param name="lr">Base learning rate.</param>
    /// <param name="shape">Parameter shape.</param>
    /// <returns>Adjusted learning rate.</returns>
    public double AdjustLearningRate(double lr, long[] shape)
    {
      if (shape.Length < 2)
      {
        return lr;
      }
      long A = shape[0];
      long B = shape[1];
      double adjusted_ratio = 0.2 * Math.Sqrt(Math.Max(A, B));
      double adjusted_lr = lr * adjusted_ratio;
      return adjusted_lr;
    }

    /// <summary>
    /// Newton–Schulz iteration to compute the “zeroth power” (an approximate orthogonalization).
    /// </summary>
    /// <param name="G">A 2D tensor.</param>
    /// <param name="steps">Number of iterations.</param>
    /// <returns>The orthogonalized tensor.</returns>
    public static Tensor ZeropowerViaNewtonSchulz5(Tensor G, int steps)
    {
      if (G.ndim != 2)
      {
        throw new ArgumentException("Tensor G must be 2D");
      }
      double a = 3.4445;
      double b = -4.7750;
      double c = 2.0315;

      // Convert to bfloat16.
      Tensor X = G.to(torch.bfloat16);
      if (G.size(0) > G.size(1))
      {
        X = X.transpose(0, 1);
      }
      // Normalize X so that its spectral norm is at most 1.
      X = X.div(X.norm().add(1e-7));
      for (int i = 0; i < steps; i++)
      {
        Tensor A = X.matmul(X.transpose(0, 1));
        Tensor B_tensor = A.mul(b).add(A.matmul(A).mul(c));
        X = X.mul(a).add(B_tensor.matmul(X));
      }
      if (G.size(0) > G.size(1))
      {
        X = X.transpose(0, 1);
      }
      return X;
    }

    /// <summary>
    /// Adds a parameter group to the optimizer.
    /// </summary>
    /// <param name="param_group">The parameter group.</param>
    public override void add_param_group(TorchSharp.Modules.ParamGroup param_group)
    {
      Options def = _defaults;
      if (param_group.Options == null)
      {
        param_group.Options = new Options();
      }
      Options opt = param_group.Options as Options;

      if (!opt.LearningRate.HasValue)
      {
        opt.LearningRate = def.LearningRate;
      }
      if (!opt.wd.HasValue)
      {
        opt.wd = def.wd;
      }
      if (!opt.momentum.HasValue)
      {
        opt.momentum = def.momentum;
      }
      if (!opt.nesterov.HasValue)
      {
        opt.nesterov = def.nesterov;
      }
      if (!opt.nsSteps.HasValue)
      {
        opt.nsSteps = def.nsSteps;
      }
      if (!opt.adamw_beta1.HasValue)
      {
        opt.adamw_beta1 = def.adamw_beta1;
      }
      if (!opt.adamw_beta2.HasValue)
      {
        opt.adamw_beta2 = def.adamw_beta2;
      }
      if (!opt.adamw_eps.HasValue)
      {
        opt.adamw_eps = def.adamw_eps;
      }
      opt.InitialLearningRate = opt.LearningRate;

      _parameter_groups.Add(param_group as MuonOptimizer.ParamGroup);

      foreach (Parameter p in param_group.Parameters)
      {
        State state = new State(p);
        // By default, we set UseMuon to false. (It is later updated in the constructor that takes separate lists.)
        state.UseMuon = false;
        _state[p.Handle] = state;
        state.Initialize(opt);
      }
    }

    /// <summary>
    /// The options class for MuonOptimizer.
    /// </summary>
    public class Options : OptimizerOptions
    {
      public double? LearningRate;
      public double? wd;
      public double? momentum;
      public bool? nesterov;
      public int? nsSteps;
      public double? adamw_beta1;
      public double? adamw_beta2;
      public double? adamw_eps;

      public double? InitialLearningRate;

      public override void LoadStateDict(OptimizerOptions source)
      {
        base.LoadStateDict(source);
        Options opts = source as Options;
        this.LearningRate = opts.LearningRate;
        this.wd = opts.wd;
        this.momentum = opts.momentum;
        this.nesterov = opts.nesterov;
        this.nsSteps = opts.nsSteps;
        this.adamw_beta1 = opts.adamw_beta1;
        this.adamw_beta2 = opts.adamw_beta2;
        this.adamw_eps = opts.adamw_eps;
      }

      public override void LoadStateDict(BinaryReader reader)
      {
        base.LoadStateDict(reader);
        this.LearningRate = reader.ReadDouble();
        this.wd = reader.ReadDouble();
        this.momentum = reader.ReadDouble();
        this.nesterov = reader.ReadBoolean();
        this.nsSteps = reader.ReadInt32();
        this.adamw_beta1 = reader.ReadDouble();
        this.adamw_beta2 = reader.ReadDouble();
        this.adamw_eps = reader.ReadDouble();
      }

      public override void SaveStateDict(BinaryWriter writer)
      {
        base.SaveStateDict(writer);
        writer.Write(this.LearningRate.Value);
        writer.Write(this.wd.Value);
        writer.Write(this.momentum.Value);
        writer.Write(this.nesterov.Value);
        writer.Write(this.nsSteps.Value);
        writer.Write(this.adamw_beta1.Value);
        writer.Write(this.adamw_beta2.Value);
        writer.Write(this.adamw_eps.Value);
      }
    }

    /// <summary>
    /// Parameter group for MuonOptimizer.
    /// </summary>
    public class ParamGroup : ParamGroup<Options>
    {
      public ParamGroup()
          : base()
      {
      }

      public ParamGroup(IEnumerable<Parameter> parameters, Options options)
          : base(parameters, options)
      {
      }

      public ParamGroup(IEnumerable<Parameter> parameters,
          double lr = 1e-3,
          double wd = 0.1,
          double momentum = 0.95,
          bool nesterov = true,
          int nsSteps = 5,
          double adamw_beta1 = 0.95,
          double adamw_beta2 = 0.95,
          double adamw_eps = 1e-8)
          : base(parameters, new Options
          {
            LearningRate = lr,
            wd = wd,
            momentum = momentum,
            nesterov = nesterov,
            nsSteps = nsSteps,
            adamw_beta1 = adamw_beta1,
            adamw_beta2 = adamw_beta2,
            adamw_eps = adamw_eps,
            InitialLearningRate = lr
          })
      {
      }
    }

    /// <summary>
    /// Internal state for each parameter.
    /// </summary>
    public class State : OptimizerState, IDisposable
    {
      public long Step;
      public Tensor MomentumBuffer;   // For Muon update.
      public Tensor AdamwMoment1;     // For AdamW backup.
      public Tensor AdamwMoment2;     // For AdamW backup.
      public bool UseMuon;

      public State(Parameter parameter)
          : base(parameter)
      {
      }

      public override void Initialize(OptimizerOptions options)
      {
        this.Step = 0;
        if (this.UseMuon)
        {
          if ((object)this._parameter != null)
          {
            this.MomentumBuffer = torch.zeros_like(this._parameter).DetachFromDisposeScope();
          }
        }
        else
        {
          if ((object)this._parameter != null)
          {
            Tensor grad = this._parameter.grad;
            if ((object)grad == null)
            {
              grad = torch.zeros_like(this._parameter);
            }
            this.AdamwMoment1 = torch.zeros_like(grad).DetachFromDisposeScope();
            this.AdamwMoment2 = torch.zeros_like(grad).DetachFromDisposeScope();
          }
        }
      }

      public override void to(Device device)
      {
        if ((object)this.MomentumBuffer != null)
        {
          this.MomentumBuffer = this.MomentumBuffer.to(device);
        }
        if ((object)this.AdamwMoment1 != null)
        {
          this.AdamwMoment1 = this.AdamwMoment1.to(device);
        }
        if ((object)this.AdamwMoment2 != null)
        {
          this.AdamwMoment2 = this.AdamwMoment2.to(device);
        }
      }

      public override void LoadStateDict(BinaryReader reader)
      {
        this.Step = reader.ReadInt64();
        if ((object)this.MomentumBuffer != null)
        {
          this.MomentumBuffer.Load(reader);
        }
        if ((object)this.AdamwMoment1 != null)
        {
          this.AdamwMoment1.Load(reader);
        }
        if ((object)this.AdamwMoment2 != null)
        {
          this.AdamwMoment2.Load(reader);
        }
        this.UseMuon = reader.ReadBoolean();
      }

      public override void SaveStateDict(BinaryWriter writer)
      {
        writer.Write(this.Step);
        if ((object)this.MomentumBuffer != null)
        {
          this.MomentumBuffer.Save(writer);
        }
        if ((object)this.AdamwMoment1 != null)
        {
          this.AdamwMoment1.Save(writer);
        }
        if ((object)this.AdamwMoment2 != null)
        {
          this.AdamwMoment2.Save(writer);
        }
        writer.Write(this.UseMuon);
      }

      public override void LoadStateDict(OptimizerState source)
      {
        State stState = source as State;
        this.Step = stState.Step;
        if ((object)this.MomentumBuffer != null)
        {
          this.MomentumBuffer.Dispose();
        }
        if ((object)this.AdamwMoment1 != null)
        {
          this.AdamwMoment1.Dispose();
        }
        if ((object)this.AdamwMoment2 != null)
        {
          this.AdamwMoment2.Dispose();
        }
        this.MomentumBuffer = stState.MomentumBuffer?.to(this._parameter.device, copy: true);
        this.AdamwMoment1 = stState.AdamwMoment1?.to(this._parameter.device, copy: true);
        this.AdamwMoment2 = stState.AdamwMoment2?.to(this._parameter.device, copy: true);
        this.UseMuon = stState.UseMuon;
      }

      public override bool ApproximatelyEquals(OptimizerState other)
      {
        State rhs = other as State;
        bool close = (this.Step == rhs.Step);
        if ((object)this.MomentumBuffer != null && (object)rhs.MomentumBuffer != null)
        {
          close = close && this.MomentumBuffer.allclose(rhs.MomentumBuffer);
        }
        if ((object)this.AdamwMoment1 != null && (object)rhs.AdamwMoment1 != null)
        {
          close = close && this.AdamwMoment1.allclose(rhs.AdamwMoment1);
        }
        if ((object)this.AdamwMoment2 != null && (object)rhs.AdamwMoment2 != null)
        {
          close = close && this.AdamwMoment2.allclose(rhs.AdamwMoment2);
        }
        close = close && (this.UseMuon == rhs.UseMuon);
        return close;
      }

      public void Dispose()
      {
        this.Dispose(true);
        GC.SuppressFinalize(this);
      }

      protected virtual void Dispose(bool disposing)
      {
        if (disposing)
        {
          if ((object)this.MomentumBuffer != null)
          {
            this.MomentumBuffer.Dispose();
          }
          if ((object)this.AdamwMoment1 != null)
          {
            this.AdamwMoment1.Dispose();
          }
          if ((object)this.AdamwMoment2 != null)
          {
            this.AdamwMoment2.Dispose();
          }
        }
      }
    }

  }
}


#if NOT
#region Using directives

using System;
using System.Collections.Generic;

using static TorchSharp.torch;
using TorchSharp;
using System.Linq;
using static TorchSharp.torch.optim;
using TorchSharp.Modules;

#endregion

namespace CeresTrain.Utils
{


  // State information per parameter.
  public class MuonParamState
  {
    public bool UseMuon { get; set; }
    public Tensor MomentumBuffer { get; set; }
    public long Step { get; set; }
    public Tensor Moment1 { get; set; }
    public Tensor Moment2 { get; set; }

    public MuonParamState()
    {
      UseMuon = false;
      MomentumBuffer = null;
      Step = 0;
      Moment1 = null;
      Moment2 = null;
    }
  }

  // A parameter group similar to PyTorch's optimizer param groups.
  public class ParamGroup
  {
    public List<Tensor> Params { get; set; } = new List<Tensor>();
    public float Lr { get; set; }
    public float WeightDecay { get; set; }
    public float Momentum { get; set; }
    public bool Nesterov { get; set; }
    public int NsSteps { get; set; }
    public (float, float) AdamwBetas { get; set; }
    public double AdamwEps { get; set; }

    public ParamGroup(float lr, float weightDecay, float momentum, bool nesterov, int nsSteps,
                      (float, float) adamwBetas, double adamwEps, List<Tensor> parameters)
    {
      Lr = lr;
      WeightDecay = weightDecay;
      Momentum = momentum;
      Nesterov = nesterov;
      NsSteps = nsSteps;
      AdamwBetas = adamwBetas;
      AdamwEps = adamwEps;
      Params = parameters;
    }
  }

  // Muon optimizer, which implements the momentUm orthogonalized by Newton-Schulz.
  public class MuonOptimizer : OptimizerHelper
  {
    private List<ParamGroup> _paramGroups = new List<ParamGroup>();
    private Dictionary<Tensor, MuonParamState> _state = new Dictionary<Tensor, MuonParamState>();


    // Constructor: splits parameters into two groups: those for Muon updates and those for AdamW.
    public MuonOptimizer(
        float lr,
        float weightDecay,
        IEnumerable<Tensor> muonParams,
        float momentum,
        bool nesterov,
        int nsSteps,
        IEnumerable<Tensor> adamwParams,
        (float, float) adamwBetas,
        float adamwEps)
      //: base(new IntPtr(1))
    {
      // Create a new list from the enumerable collections.
      List<Tensor> muonParameters = new(muonParams);
      List<Tensor> adamwParameters = new();
      if (adamwParams != null)
      {
        adamwParameters = new List<Tensor>(adamwParams);
      }

      // Combine parameters.
      List<Tensor> allParameters = new List<Tensor>();
      allParameters.AddRange(muonParameters);
      allParameters.AddRange(adamwParameters);

      // Create a single parameter group. In practice you might want separate groups.
      ParamGroup group = new ParamGroup(
          lr: lr,
          weightDecay: weightDecay,
          momentum: momentum,
          nesterov: nesterov,
          nsSteps: nsSteps,
          adamwBetas: adamwBetas,
          adamwEps: adamwEps,
          parameters: allParameters);
      _paramGroups.Add(group);

      // Initialize state for each parameter.
      foreach (Tensor p in muonParameters)
      {
        // Here we assume muonParams are 2D. In practice, check p.dim().
        MuonParamState state = new MuonParamState();
        state.UseMuon = true;
        _state[p] = state;
      }

      foreach (Tensor p in adamwParameters)
      {
        MuonParamState state = new MuonParamState();
        state.UseMuon = false;
        _state[p] = state;
      }
    }


    // A sample implementation of ILearningRateController.
    public class ParameterGroupController : optim.ILearningRateController
    {
      // List of parameters in this group.
      public IList<Tensor> Parameters { get; private set; }

      double initialLearningRate;

      // The learning rate associated with this parameter group.
      double _learningRate;
      double ILearningRateController.LearningRate 
      {
        get => _learningRate;
        set => _learningRate = value;
      }
      double ILearningRateController.InitialLearningRate 
      {
        get => initialLearningRate;
        set => initialLearningRate = value;
      }


      // Constructor accepts a list of parameters and the learning rate.
      public ParameterGroupController(IList<Tensor> parameters/*, double learningRate*/)
      {
        Parameters = parameters;
//        LearningRate = learningRate;
      }

      // Optionally, you could add methods to update the learning rate if needed.
      public void SetLearningRate(double lr)
      {
//        LearningRate = lr;
      }
    }

    public override IEnumerable<optim.ILearningRateController> ParamGroups
    {
      get
      {
        var muonParams = _state.Keys.Where(p => _state[p].UseMuon).ToList();
        var adamwParams = _state.Keys.Where(p => !_state[p].UseMuon).ToList();
        yield return new ParameterGroupController(muonParams);
        yield return new ParameterGroupController(adamwParams);
      }

    }


    // Adjust learning rate for muon update based on the parameter's shape.
    private double AdjustLrForMuon(double lr, long[] paramShape)
    {
      long A = paramShape[0];
      long B = paramShape[1];
      double adjustedRatio = 0.2 * Math.Sqrt(Math.Max(A, B));
      double adjustedLr = lr * adjustedRatio;
      return adjustedLr;
    }

    // Performs a single optimization step.
    public Tensor step(Func<Tensor> closure = null)
    {
      Tensor loss = null;
      if (closure != null)
      {
        using (torch.no_grad())
        {
          loss = closure();
        }
      }

      foreach (ParamGroup group in _paramGroups)
      {
        // First, process parameters that use Muon.
        List<Tensor> muonParameters = new List<Tensor>();
        foreach (Tensor p in group.Params)
        {
          if (_state[p].UseMuon)
          {
            muonParameters.Add(p);
          }
        }

        foreach (Tensor p in muonParameters)
        {
          Tensor g = p.grad;
          if (g is null)
          {
            continue;
          }
          if (g.dim() > 2)
          {
            // Reshape g to 2D: (first dimension, rest flattened)
            long firstDim = g.shape[0];
            long rest = 1;
            for (int d = 1; d < g.dim(); d++)
            {
              rest *= (int)g.shape[d];
            }
            g = g.view(new long[] { firstDim, rest });
          }

          MuonParamState state = _state[p];
          if ((object)state.MomentumBuffer == null)
          {
            state.MomentumBuffer = torch.zeros_like(g);
          }

          // Update momentum buffer: buffer = momentum * buffer + g
          state.MomentumBuffer = state.MomentumBuffer.mul(group.Momentum).add(g);

          if (group.Nesterov)
          {
            g = g.add(state.MomentumBuffer.mul(group.Momentum));
          }
          else
          {
            g = state.MomentumBuffer;
          }

          Tensor u = ZeropowerViaNewtonSchulz5(g, group.NsSteps);
          double adjustedLr = AdjustLrForMuon(group.Lr, p.shape);

          using (torch.no_grad())
          {

            // Apply weight decay.
            if (group.WeightDecay > 0)
            {
              p.mul_(1 - group.Lr * group.WeightDecay);
            }

            // Apply update.
            p.add_(u, -adjustedLr);
          }
        }

        // Next, process parameters that use AdamW backup.
        List<Tensor> adamwParameters = new List<Tensor>();
        foreach (Tensor p in group.Params)
        {
          if (!_state[p].UseMuon)
          {
            adamwParameters.Add(p);
          }
        }

        foreach (Tensor p in adamwParameters)
        {
          Tensor g = p.grad;
          if (g is null)
          {
            continue;
          }

          MuonParamState state = _state[p];
          if (state.Step == 0)
          {
            state.Moment1 = torch.zeros_like(g);
            state.Moment2 = torch.zeros_like(g);
          }

          state.Step += 1;
          long step = state.Step;
          // Update biased first and second moment estimates using linear interpolation.
          state.Moment1 = state.Moment1.lerp(g, 1 - group.AdamwBetas.Item1);
          state.Moment2 = state.Moment2.lerp(g.pow(2), 1 - group.AdamwBetas.Item2);

          Tensor gHat = state.Moment1.div((state.Moment2.sqrt()).add(group.AdamwEps));
          double biasCorrection1 = 1 - Math.Pow(group.AdamwBetas.Item1, step);
          double biasCorrection2 = 1 - Math.Pow(group.AdamwBetas.Item2, step);
          double scale = biasCorrection1 / Math.Sqrt(biasCorrection2);

          p.mul_(1 - group.Lr * group.WeightDecay);
          p.add_(gHat, -group.Lr / scale);
        }
      }

      return loss;
    }


    // Computes the zeroth power / orthogonalization of G using the Newton-Schulz iteration.
    // This function uses a quintic iteration with fixed coefficients.
    static Tensor ZeropowerViaNewtonSchulz5(Tensor G, int steps)
    {
      if (G.dim() != 2)
      {
        throw new ArgumentException("G must be a 2D tensor.");
      }

      // Coefficients
      double a = 3.4445;
      double b = -4.7750;
      double c = 2.0315;

      // Convert to bfloat16 (if supported) and assign to X.
      Tensor X = G.to(ScalarType.BFloat16);
      if (G.shape[0] > G.shape[1])
      {
        X = X.transpose(0, 1);
      }

      // Ensure spectral norm is at most 1.
      Tensor norm = X.norm() + 1e-7;
      X = X.div(norm);

      // Perform the NS iterations.
      for (int i = 0; i < steps; i++)
      {
        Tensor A = X.matmul(X.transpose(0, 1));

        // Compute B = b * A + c * (A.matmul(A))
        Tensor B = A.mul(b).add(A.matmul(A).mul(c));

        X = X.mul(a).add(B.matmul(X));
      }

      if (G.shape[0] > G.shape[1])
      {
        X = X.transpose(0, 1);
      }

      return X;
    }


    public static void Test()
    {
      // Test NewtonSchulz.ZeropowerViaNewtonSchulz5 with a dummy tensor.
      Tensor G = torch.randn([4, 4], device: torch.CPU);
      int steps = 5;
      Tensor result = ZeropowerViaNewtonSchulz5(G, steps);
      Console.WriteLine("Result of ZeropowerViaNewtonSchulz5:");
      Console.WriteLine(result.ToString());

      // Create dummy parameters (with gradients) for the Muon optimizer.
      List<Tensor> parameters = new List<Tensor>();
      Tensor param1 = torch.randn([3, 3], requires_grad: true, device: torch.CPU);
      Tensor param2 = torch.randn([4, 4], requires_grad: true, device: torch.CPU);
      parameters.Add(param1);
      parameters.Add(param2);

      // Instantiate the Muon optimizer.
      MuonOptimizer muonOptimizer = new(0.02f, 0.01f,
                                         parameters,
                                         momentum: 0.95f, nesterov: true, nsSteps: 5,
                                         new List<Tensor>(), (0.95f, 0.95f), 1E-8f);

      // Perform a dummy backward pass.
      Tensor loss = param1.sum() + param2.sum();
      loss.backward();

      // Take an optimization step.
      muonOptimizer.step();
      Console.WriteLine("Muon optimizer step completed.");
    }
  }
}

#endif