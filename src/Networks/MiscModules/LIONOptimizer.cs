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

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

#endregion

#region Using directives

using System.Collections.Generic;
using System.IO;
using System;

using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim;

#endregion

namespace CeresTrain.Networks.MiscModules
{
  /// <summary>
  /// LION optimizer.
  /// 
  /// Algorithm from : https://arxiv.org/pdf/2302.06675.pdf
  /// Code based on : https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lion.py
  /// See also for optimized (Triton kernel) Pytorch implementation: https://github.com/lucidrains/lion-pytorch/tree/main/lion_pytorch
  /// </summary>
  public class LIONOptimizer : OptimizerHelper//, IMomentum
  {
    public LIONOptimizer(IEnumerable<Parameter> parameters,
                         float lr, float beta1 = 0.0f, float beta2 = 0,
                         float weight_decay = 0, bool maximize = false)
        : this(new LIONParamGroup[] { new LIONParamGroup { Parameters = parameters } }, lr, beta1, beta2, weight_decay, maximize)
    {
    }

    public LIONOptimizer(IEnumerable<LIONParamGroup> parameters,
                        float lr, float beta1 = 0.0f, float beta2 = 0,
                        float weight_decay = 0, bool maximize = false)
    {
      if (lr < 0.0)
      {
        throw new ArgumentException($"Invalid learning rate: {lr}");
      }
      if (weight_decay < 0.0)
      {
        throw new ArgumentException($"Invalid weight_decay value: {weight_decay}");
      }

      LIONOptions options = new LIONOptions
      {
        LearningRate = lr,
        InitialLearningRate = lr,
        beta1 = beta1,
        beta2 = beta2,
        weight_decay = weight_decay,
        maximize = maximize,
      };

      _defaults = options;
      _parameter_groups = new List<ParamGroup>();

      foreach (LIONParamGroup g in parameters)
      {
        add_param_group(g);
      }
    }

    void update_fn(Parameter p, Tensor grad, Tensor exp_avg, float lr, float wd, float beta1, float beta2)
    {
      // stepweight decay
      if (wd != 0.0)
      {
        p.mul_(1 - lr * wd);
      }

      // weight update
      var update = exp_avg.clone().mul_(beta1).add(grad, 1 - beta1).sign_();
      p.add_(update, -lr);

      // decay the momentum running average coefficient
      exp_avg.mul_(beta2).add_(grad, 1 - beta2);
    }


    public override Tensor step(Func<Tensor> closure = null)
    {
      return _step<LIONParamGroup>(group =>
      {
        LIONOptions options = group.Options;
        float beta1 = options.beta1;
        float beta2 = options.beta2;
        float weight_decay = options.weight_decay.Value;
        bool maximize = options.maximize.Value;
        float lr = (float)options.LearningRate.Value;

        Tensor loss;
        if (closure != null)
        {
          using (enable_grad())
          {
            loss = closure();
          }
        }

#if NOT
      for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], 
                group['weight_decay'], *group['betas'], 
                self.state[p]

                // init state - exponential moving average of gradient values
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                update_fn(p, grad, exp_avg, lr, wd, beta1, beta2

        return loss
#endif
        foreach (Parameter param in group.Parameters)
        {
          Tensor grad = param.grad();
          if (grad is null)
          {
            continue;
          }

          State state = (State)_state[param.Handle];

          // init state - exponential moving average of gradient values
          //if (state.exp_avg.size().Length == 0)
          Tensor exp_avg = state.exp_avg;
          if (exp_avg is null || exp_avg is null || exp_avg.size().Length == 0)
          {
            exp_avg = zeros_like(param).MoveToOuterDisposeScope();
          }


          update_fn(param, grad, exp_avg, lr, weight_decay, beta1, beta2);
        }
      }, closure);
    }

    protected override void Dispose(bool disposing)
    {
      base.Dispose(disposing);
      _state.Clear();
    }

    public class State : OptimizerState, IDisposable
    {
      public Tensor exp_avg = tensor(0);

      public State(Parameter parameter) : base(parameter)
      {
      }

      public void Dispose()
      {
        exp_avg.Dispose();
      }

      public override void to(Device device)
      {
        if (exp_avg is not null)
        {
          exp_avg.to(device);
        }
      }

      public override void LoadStateDict(BinaryReader reader)
      {
        bool hasMomentumBuffer = reader.ReadBoolean();
        if (hasMomentumBuffer)
        {
          int ndim = reader.ReadInt32();
          long[] shape = new long[ndim];

          for (int i = 0; i < ndim; i++)
          {
            shape[i] = reader.ReadInt64();
          }

          if (exp_avg is null)
          {
            exp_avg = empty(shape).DetachFromDisposeScope();
          }
          exp_avg.Load(reader);
        }
        else
        {
          if (exp_avg is not null)
          {
            exp_avg.Dispose();
          }
          exp_avg = null;
        }
      }

      public override void SaveStateDict(BinaryWriter writer)
      {
        if (exp_avg is not null)
        {
          writer.Write(true);
          writer.Write(exp_avg.shape.Length);
          for (int i = 0; i < exp_avg.shape.Length; i++)
          {
            writer.Write(exp_avg.shape[i]);
          }
          exp_avg.Save(writer);
        }
        else
        {
          writer.Write(false);
        }
      }

      public override void LoadStateDict(OptimizerState source)
      {
        State st_state = source as State;
        if (exp_avg is not null)
        {
          exp_avg.Dispose();
        }
        exp_avg = st_state.exp_avg;
      }

      public override bool ApproximatelyEquals(OptimizerState other)
      {
        State rhs = other as State;
        return rhs is not null && (exp_avg is null || exp_avg.allclose(rhs.exp_avg));
      }

      public override void Initialize(OptimizerOptions options)
      {
        throw new NotImplementedException();
      }
    }

    public override void add_param_group(ParamGroup param_group)
    {
      LIONOptions def = _defaults as LIONOptions;
      if (param_group.Options is null)
      {
        param_group.Options = new LIONOptions();
      }

      LIONOptions opt = param_group.Options as LIONOptions;

      // Make sure all the options are set.
      opt.LearningRate = def.LearningRate;
      opt.lr = def.lr;
      opt.beta1 = def.beta1;
      opt.beta2 = def.beta2;
      if (!opt.weight_decay.HasValue) opt.weight_decay = def.weight_decay;
      if (!opt.maximize.HasValue) opt.maximize = def.maximize;

      opt.InitialLearningRate = opt.LearningRate.Value;

      _parameter_groups.Add(param_group);

      foreach (var p in param_group.Parameters)
      {
        var state = new State(p);
        _state[p.Handle] = state;
      }
    }

    public class LIONOptions : OptimizerOptions
    {
      public float lr;
      public float beta1;
      public float beta2;
      public float? weight_decay;
      public bool? maximize;

      public override void LoadStateDict(BinaryReader reader)
      {
        base.LoadStateDict(reader);
        lr = (float)reader.ReadDouble();
        beta1 = (float)reader.ReadDouble();
        beta2 = (float)reader.ReadDouble();
        weight_decay = (float)reader.ReadDouble();
        maximize = reader.ReadBoolean();
      }

      public override void LoadStateDict(OptimizerOptions source)
      {
        base.LoadStateDict(source);
        var opts = source as LIONOptions;
        lr = opts.lr;
        beta1 = opts.beta1;
        beta1 = opts.beta1;
        weight_decay = opts.weight_decay;
        maximize = opts.maximize;
      }

      public override void SaveStateDict(BinaryWriter writer)
      {
        base.SaveStateDict(writer);
        writer.Write(lr);
        writer.Write(beta1);
        writer.Write(beta2);
        writer.Write(weight_decay.Value);
        writer.Write(maximize.Value);
      }
    }

    public class LIONParamGroup : ParamGroup<LIONOptions> //DJE, IMomentum
    {
      public LIONParamGroup() { }

      public LIONParamGroup(IEnumerable<Parameter> parameters, LIONOptions options) : base(parameters, options) { }

      public LIONParamGroup(IEnumerable<Parameter> parameters, float lr = 1e-3f, float beta1 = 0.0f, float beta2 = 0, float weight_decay = 0, bool maximize = false)
          : base(parameters, new LIONOptions { LearningRate = lr, beta1 = beta1, beta2 = beta2, weight_decay = weight_decay, maximize = maximize })
      {
      }
    }

  }
}
