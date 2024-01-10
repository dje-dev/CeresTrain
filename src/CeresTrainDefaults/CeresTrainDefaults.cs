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

using CeresTrain.Trainer;
using CeresTrain.Networks.Transformer;

#endregion

namespace CeresTrain.CeresTrainDefaults
{
  /// <summary>
  /// Static class containing default values for CeresTrain.
  /// This includes the default ConfigTraining (DEFAULT_CONFIG_TRAINING)
  /// used to as the starting point for initialized training configuration
  /// (as created by the "init" command).
  /// </summary>
  public static class CeresTrainDefault
  {
    /// <summary>
    /// Default ConfigTraining used by the init command.
    /// 
    /// Customize as desired.
    /// 
    /// Alternately, the JSON files containing a configuration can be directly edited subsequent to initialization.
    /// </summary>
    public static ConfigTraining DEFAULT_CONFIG_TRAINING = new ConfigTraining() with
    {
      // Configuration entries relating to execution environment (e.g. devices).
      ExecConfig = new ConfigNetExecution() with
      {
        ID = "Test1"
      },


      // Configuration entries relating to the DataSet used for training.
      DataConfig = new ConfigData() with
      {

      },

      NetDefConfig = new NetTransformerDef(192, 6, 8, 1, NetTransformerDef.TransformerFeatures.Smolgen) with
      {
      },

      OptConfig = new ConfigOptimization() with
      {
        LearningRateBase = 5E-4f,
        LossValueMultiplier = 0.5f,
        BatchSizeForwardPass = 2048,
        BatchSizeBackwardPass = 2048,
        PyTorchCompileMode = "max-autotune"
      },


      // Configuration entries relating to monitoring of the training process.
      MonitoringConfig = new ConfigMonitoring() with
      {
      },
    };

  }
}
