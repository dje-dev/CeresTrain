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
using System.IO;
using System.Text.Json.Serialization;
using System.Text.Json;

#endregion

namespace CeresTrain.Trainer
{
  /// <summary>
  /// Summary information relating to a training run.
  /// </summary>
  public readonly record struct TrainingResultSummary
  {
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="executionCoordinatorHost"></param>
    /// <param name="trainingHost"></param>
    /// <param name="configName"></param>
    /// <param name="trainingEndTime"></param>
    /// <param name="trainingEndStatus"></param>
    /// <param name="numParameters"></param>
    /// <param name="trainingTime"></param>
    /// <param name="numTrainingPositions"></param>
    /// <param name="trainingLogFileName"></param>
    /// <param name="finalTorchscriptFileName"></param>
    /// <param name="finalONNXFileName"></param>
    /// <exception cref="ArgumentNullException"></exception>
    public TrainingResultSummary(string executionCoordinatorHost, string trainingHost, string configName, 
                                 DateTime trainingEndTime, string trainingEndStatus,
                                 long numParameters,
                                 TimeSpan trainingTime, long numTrainingPositions,
                                 TrainingLossSummary lossSummary,
                                 string trainingLogFileName,
                                 string finalTorchscriptFileName, string finalONNXFileName)
    {
      LaunchTrainingHost = executionCoordinatorHost;
      TrainingHost = trainingHost;
      ConfigID = configName;
      TrainingEndTime = trainingEndTime;
      TrainingEndStatus = trainingEndStatus;
      NumParameters = numParameters;
      TrainingTime = trainingTime;
      NumTrainingPositions = numTrainingPositions;
      LossSummary = lossSummary;
      TrainingLogFileName = trainingLogFileName;
      TorchscriptFileName = finalTorchscriptFileName;
      ONNXFileName = finalONNXFileName;
    }


    /// <summary>
    /// JSON serialization constructor.
    /// </summary>
    [JsonConstructor]
    public TrainingResultSummary()
    {

    }


    /// <summary>
    /// Host on which launch/coordination of training took place.
    /// </summary>
    public readonly string LaunchTrainingHost { get; init; }


    /// <summary>
    /// Host on which training took place.
    /// </summary>
    public readonly string TrainingHost { get; init; }

    /// <summary>
    /// ID of the ConfigTraining used for training.
    /// </summary>
    public readonly string ConfigID { get; init; }

    /// <summary>
    /// Date/Time on which training ended.
    /// </summary>
    public readonly DateTime TrainingEndTime { get; init; }

    /// <summary>
    /// Status code indicating how training ended.
    /// </summary>
    public readonly string TrainingEndStatus { get; init; }

    /// <summary>
    /// Number of trainable parameters in the neural network.
    /// </summary>
    public readonly long NumParameters { get; init; }

    /// <summary>
    /// Elapsed time during training.
    /// </summary>
    public readonly TimeSpan TrainingTime { get; init; }

    /// <summary>
    /// Number of positions seen by the neural network during training.
    /// </summary>
    public readonly long NumTrainingPositions { get; init; }

    /// <summary>
    /// Summary of losses/statistics at end of training.
    /// </summary>
    public readonly TrainingLossSummary LossSummary { get; init; }

    /// <summary>
    /// Name of the file containing training log (all lines sent to stdout/stderr during training).
    /// </summary>
    public readonly string TrainingLogFileName { get; init; }

    /// <summary>
    /// Name of the model file containing the neural network (last saved).
    /// </summary>
    public readonly string TorchscriptFileName { get; init; }

    /// <summary>
    /// Name of the model file containing the neural network (having lowest loss during training).
    /// </summary>
    public readonly string ONNXFileName { get; init; }


    /// <summary>
    /// Loads and returns a summary from a JSON file.
    /// </summary>
    public static TrainingResultSummary LoadFromFile(string baseOutputDir, string configID)
    {
      string resultsFileName = Path.Combine(baseOutputDir, configID);
      if (!File.Exists(resultsFileName))
      {
        throw new ArgumentException($"No such file: {resultsFileName}");
      }

      string jsonString = File.ReadAllText(resultsFileName);
      JsonSerializerOptions options = new JsonSerializerOptions() { AllowTrailingCommas = true };
      return JsonSerializer.Deserialize<TrainingResultSummary>(jsonString, options);
    }


    /// <summary>
    /// Writes summary as a JSON file.
    /// </summary>
    /// <param name="baseOutputDir"></param>
    public string WriteJSON(string baseOutputDir, string configID)
    {
      Directory.CreateDirectory(baseOutputDir);
      string outFN = Path.Combine(baseOutputDir, configID + "_results.json");

      JsonSerializerOptions options = new JsonSerializerOptions
      {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() },
      };

      File.WriteAllText(outFN, JsonSerializer.Serialize(this, options));
      return outFN;

    }
  }
}
