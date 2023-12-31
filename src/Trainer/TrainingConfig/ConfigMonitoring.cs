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
using System.Text.Json.Serialization;
using Ceres.Chess;
using Ceres.Chess.NNEvaluators.Defs;
using CeresTrain.Networks;
using CeresTrain.NNEvaluators;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Defines the set of monitoring features to be enabled during training.
  /// </summary>
  public readonly record struct ConfigMonitoring
  {
    /// <summary>
    /// Constructor.
    /// </summary>
    public ConfigMonitoring()
    {
    }

    /// <summary>
    /// Maximum interval (in seconds) at which CustomMonitoringCallback is called.
    /// </summary>
    public float CustomMonitoringIntervalSeconds { get; init; } = 600f;

    /// <summary>
    /// Option callback function which is called periodically.
    /// Returns summary and detail strings which will be output to Console and a text file, respectively.
    /// </summary>
    [JsonIgnore]
    public Func<CeresNeuralNet, ConfigTraining, NNEvaluatorDef, NNEvaluatorTorchsharp, (string outputConsole, string outputFile)> CustomMonitoringCallback { get; init; }

    /// <summary>
    /// Definition of the network against which the training network will be compared.
    /// </summary>
    public NNEvaluatorNetDef CompareNetDef { get; init; }

    /// <summary>
    /// Definition of the device on which the comparison network will be executed.
    /// </summary>
    public NNEvaluatorDeviceDef CompareDeviceDef { get; init; } = new NNEvaluatorDeviceDef(NNDeviceType.GPU, 0);

    /// <summary>
    /// Name of file containing openings to be used for testing the training network against the comparison network.
    /// </summary>
    public string CompareNetOpeningsFileName { get; init; }

    /// <summary>
    /// Number of game pairs to play between the training network and the comparison network (policy head test).
    /// </summary>
    public int CompareNetTestPolicyNumGamePairs { get; init; }

    /// <summary>
    /// Number of game pairs to play between the training network and the comparison network (value head test).
    /// </summary>
    public int CompareNetTestValueNumGamePairs { get; init; }

    /// <summary>
    /// Name of the file containing the EPD test suite positions to be used for testing the training network.
    /// </summary>
    public readonly string TestSuiteFileName { get; init; }

    /// <summary>
    /// Maxiumum number of positions to be used from the test suite.
    /// </summary>
    public readonly int TestSuiteMaxPositions { get; init; } = 100;

    /// <summary>
    /// Search limit to be used with each test suite position.
    /// </summary>
    public readonly SearchLimit TestSuiteSearchLimit { get; init; } = SearchLimit.NodesPerMove(100);

    /// <summary>
    /// If the internal activation statistics should be dumped.
    /// </summary>
    public bool DumpInternalActivationStatistics { get; init; }

    /// <summary>
    /// If the internal parameter statistics should be dumped.
    /// </summary>
    public bool DumpInternalParameterStatistics { get; init; }
  }
}
