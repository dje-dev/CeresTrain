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

using System.Linq;
using System;

using static TorchSharp.torch;

using CeresTrain.NNIntrospection;
using CeresTrain.Utils;
using System.Text.Json.Serialization;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Method of execution of the neural network.
  /// </summary>
  public enum NNEvaluatorInferenceEngine
  {
    /// <summary>
    /// Executed via Torchsharp code in C# (weights loaded from Torchscript file).
    /// </summary>
    CSharpViaTorchscript,

    /// <summary>
    /// Executed by using Torch to execute a specified Torchscript file.
    /// </summary>
    TorchViaTorchscript,
  }


  /// <summary>
  /// Definition of all the parameters relating to execution of a CeresNeuralNetwork
  /// and optionally the path to a Torchscript file which contains the parameters
  /// (possibly also with a second Torchscript file to be be averaged in).
  /// </summary>
  public readonly record struct ConfigNetExecution
  {
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="deviceType">torch.Device type such as "cuda" or "cpu"</param>
    /// <param name="deviceIDs">array of device IDs to be used</param>
    /// <param name="dataType"></param>
    /// <param name="networkFileName1"></param>
    /// <param name="networkFileName2"></param>
    /// <param name="engineType"></param>
    /// <param name="activationMonitoringSkipCount"></param>
    public ConfigNetExecution(string id, 
                              string deviceType, 
                              int[] deviceIDs = null,
                              ScalarType dataType = ScalarType.BFloat16,
                              string networkFileName1 = null,
                              string networkFileName2 = null,
                              NNEvaluatorInferenceEngine engineType = NNEvaluatorInferenceEngine.CSharpViaTorchscript,
                              int activationMonitoringSkipCount = 0)
    {
      ArgumentNullException.ThrowIfNull(deviceIDs);

      ID = id;
      TorchSharpUtils.ValidatedDevice(new Device(deviceType, deviceIDs[0]));
      DeviceType = deviceType;
      DeviceIDs = deviceIDs;
      DataType = dataType;
      SaveNetwork1FileName = networkFileName1;
      SaveNetwork2FileName = networkFileName2;
      EngineType = engineType;
      ActivationMonitorDumpSkipCount = activationMonitoringSkipCount;
    }

    /// <summary>
    /// Returns the device 
    /// </summary>
    [JsonIgnore]
    public Device Device
    {
      get
      {
        if (DeviceIDs.Length > 1)
        {
          throw new Exception("DeviceIDs.Length > 1, not support in C# backend (only Python backend)");
        }

        return new Device(DeviceType, DeviceIDs[0]);
      }
    }

    /// <summary>
    /// Default constructor for deserialization.
    /// </summary>
    [JsonConstructorAttribute]
    public ConfigNetExecution()
    {

    }


    /// <summary>
    /// Identifying string.
    /// </summary>
    public readonly string ID { get; init; } = "TestModel";

    /// <summary>
    /// Type of device on which to run the model.
    /// </summary>
    public readonly string DeviceType { get; init; } = "cuda";

    /// <summary>
    /// Set of device IDs when training on multiple GPUs (only for Python backend).
    /// </summary>
    public readonly int[] DeviceIDs { get; init; } = [0];

    /// <summary>
    /// Data type to be used for weights/parameters of the model.
    /// </summary>
    public readonly ScalarType DataType { get; init; } = ScalarType.BFloat16;

    /// <summary>
    /// If training session should be run under specified host docker launch command.
    /// </summary>
    public readonly bool RunInDocker { get; init; } = false;

    /// <summary>
    /// If dropout is used, this is the dropout probability as a fraction (e.g. 0.1 for 10% dropped out).
    /// </summary>
    public readonly float DropoutRate { get; init; }


    /// <summary>
    /// If true, dropout is used during inference (not just during training).
    /// </summary>
    public readonly bool DropoutDuringInference { get; init; } = false;

    /// <summary>
    /// Type of inference engine to use.
    /// </summary>
    public readonly NNEvaluatorInferenceEngine EngineType { get; init; }

    /// <summary>
    /// Path to file containing primary model parameters (either Torchscript format or Torchsharp format depending on EngineType).
    /// </summary>
    public readonly string SaveNetwork1FileName { get; init; }

    /// <summary>
    /// Path to file containing secondary model parameters (either Torchscript format or Torchsharp format depending on EngineType).
    /// </summary>
    public readonly string SaveNetwork2FileName { get; init; }

    /// <summary>
    /// Number of batch evaluations to skip between activation monitoring dumps.
    /// </summary>
    public readonly int ActivationMonitorDumpSkipCount { get; init; }

    /// <summary>
    /// Type of supplementary activation monitoring to use.
    /// </summary>
    public readonly NNLayerMonitor.SupplementaryStatType SupplementaryStat { get; init; } = NNLayerMonitor.SupplementaryStatType.None;

    /// <summary>
    /// If the intrinsic dimensionality (using TwoNN) should be tracked for the final layer.
    /// </summary>
    public readonly bool TrackFinalLayerIntrinsicDimensionality { get; init; } = false;

    // TODO: Break out some of these properties such as TrackActivationStats into a ConfigInferencingOptions record
    public readonly bool MonitorActivationStats { get; init; } = false;

    /// <summary>
    /// Reserved value for debugging/experimentation to turn on a possible ad hoc test/diagnostic feature.
    /// </summary>
    public readonly bool TestFlag { get; init; } = false;

    /// <summary>
    /// Reserved value used for debugging/experimentation to turn on a possible ad hoc test/diagnostic feature.
    /// </summary>
    public readonly float TestValue { get; init; } = 0;

  }
}
