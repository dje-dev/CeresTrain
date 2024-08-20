# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

import os
import json

"""
Global constants.
"""
NUM_TOKENS_INPUT = 64 # Raw input number of tokens
NUM_TOKENS_NET = 64 # Number of tokens used by net
NUM_INPUT_BYTES_PER_SQUARE = 137 # Raw input width per token


def read_config(file_path):
  """Reads a JSON configuration file and returns a dictionary."""
  with open(file_path, 'r') as file:
      return json.load(file)

class Configuration:
  """
    Initializes the Configuration object by loading various configuration settings from JSON files.

    This method reads multiple configuration files corresponding to different aspects of the training 
    process (execution, data handling, optimization, and network definition). Each configuration file 
    is identified by the provided 'id'. The method loads these settings into the object's attributes 
    for easy access throughout the program.

    Parameters:
        config_files_dir (str): The directory path where configuration files are stored.
        id (str): A unique identifier used to specify which configuration files to load. 
                    This ID is appended to each config file's base name.
  """
    
  def __init__(self, config_files_dir : str, id : str):
    self.id = id

    print('READING CONFIG FILES FOR TRAINING RUN", id, "FROM DIRECTORY', config_files_dir)
    config_data_file = os.path.join(config_files_dir, id + '_ceres_data.json')
    config_exec_file = os.path.join(config_files_dir, id + '_ceres_exec.json')
    config_opt_file = os.path.join(config_files_dir,  id + '_ceres_opt.json')
    config_net_def_file = os.path.join(config_files_dir, id + '_ceres_net.json')

    config_exec = read_config(config_exec_file)
    config_data = read_config(config_data_file)
    config_opt = read_config(config_opt_file)
    config_net_def = read_config(config_net_def_file)

    # Initialize class members from config_data
    self.Data_SourceType = config_data.get('SourceType', 'DirectFromPositionGenerator')
    self.Data_PositionGenerator = config_data.get('PositionGenerator', {})
    self.Data_TrainingFilesDirectory = config_data.get('TrainingFilesDirectory', None)
    self.Data_NumTPGFilesToSkip = config_data.get('Data_NumTPGFilesToSkip', 0)
    self.Data_FractionQ = config_data.get('FractionQ', 0.0)
    self.Data_WDLLabelSmoothing = config_data.get('WDLLabelSmoothing', 0.0)

    # Initialize class members from config_exec
    self.Exec_ID = config_exec.get('ID', 'TEST')
    self.Exec_DeviceType = config_exec.get('DeviceType', 'CUDA')
    self.Exec_DeviceIDs = config_exec.get('DeviceIDs', [0])
    self.Exec_DataType = config_exec.get('DataType', 'BFloat16')
    self.Exec_UseFP8 = config_exec.get('UseFP8', False)
    self.Exec_DropoutRate = config_exec.get('DropoutRate', 0)
    self.Exec_DropoutDuringInference = config_exec.get('DropoutDuringInference', False)
    self.Exec_EngineType = config_exec.get('EngineType', 0)
    self.Exec_SaveNetwork1FileName = config_exec.get('SaveNetwork1FileName', None)
    self.Exec_SaveNetwork2FileName = config_exec.get('SaveNetwork2FileName', None)
    self.Exec_ActivationMonitorDumpSkipCount = config_exec.get('ActivationMonitorDumpSkipCount', 0)
    self.Exec_SupplementaryStat = config_exec.get('SupplementaryStat', None)
    self.Exec_TrackFinalLayerIntrinsicDimensionality = config_exec.get('TrackFinalLayerIntrinsicDimensionality', False)
    self.Exec_MonitorActivationStats = config_exec.get('MonitorActivationStats', False)
    self.Exec_ExportOnly = config_exec.get('ExportOnly', False)
    self.Exec_TestFlag = config_exec.get('TestFlag', False)
    self.Exec_TestValue = config_exec.get('TestValue', 0)

    # Initialize class members from config_opt
    self.Opt_NumTrainingPositions = config_opt.get('NumTrainingPositions', 10_000_000)
    self.Opt_BatchSizeForwardPass = config_opt.get('BatchSizeForwardPass', 2048)
    self.Opt_BatchSizeBackwardPass = config_opt.get('BatchSizeBackwardPass', 2048)
    self.Opt_Optimizer = config_opt.get('Optimizer', 'AdamW')
    self.Opt_CheckpointResumeFromFileName = config_opt.get('CheckpointResumeFromFileName')
    self.Opt_CheckpointFrequencyNumPositions = config_opt.get('CheckpointFrequencyNumPositions', 200_000_000)
    self.Opt_PyTorchCompileMode = config_opt.get('PyTorchCompileMode', "max-autotune")
    self.Opt_WeightDecay = config_opt.get('WeightDecay', 0.01)
    self.Opt_LearningRateBase = config_opt.get('LearningRateBase', 0.0005)
    self.Opt_LRBeginDecayAtFractionComplete = config_opt.get('LRBeginDecayAtFractionComplete', 0.25)
    self.Opt_Beta1 = config_opt.get('Beta1', 0.90)
    self.Opt_Beta2 = config_opt.get('Beta2', 0.98)
    self.Opt_GradientClipLevel = config_opt.get('GradientClipLevel', 1.0)
    self.Opt_LossValueMultiplier = config_opt.get('LossValueMultiplier', 0.5)
    self.Opt_LossValue2Multiplier = config_opt.get('LossValue2Multiplier', 0.0)
    self.Opt_LossValueDMultiplier = config_opt.get('LossValueDMultiplier', 0.0)
    self.Opt_LossValue2DMultiplier = config_opt.get('LossValue2DMultiplier', 0.0)
    self.Opt_LossUncertaintyPolicyMultiplier = config_opt.get('LossUncertaintyPolicyMultiplier', 0.0)
    self.Opt_LossActionMultiplier = config_opt.get('LossActionMultiplier', 0.0)
    self.Opt_LossActionUncertaintyMultiplier = config_opt.get('LossActionUncertaintyMultiplier', 0.0)
    self.Opt_LossQDeviationMultiplier = config_opt.get('LossQDeviationMultiplier', 0.0)    
    self.Opt_LossPolicyMultiplier = config_opt.get('LossPolicyMultiplier', 1)
    self.Opt_LossMLHMultiplier = config_opt.get('LossMLHMultiplier', 0)
    self.Opt_LossUNCMultiplier = config_opt.get('LossUNCMultiplier', 0)
    self.Opt_TestValue = config_opt.get('TestValue', 0)

    # Initialize class members from config_net_def
    self.NetDef_ModelDim = config_net_def.get('ModelDim', 256)
    self.NetDef_NumLayers = config_net_def.get('NumLayers', 8)
    self.NetDef_NumHeads = config_net_def.get('NumHeads', 8)
    self.NetDef_DualAttentionMode = config_net_def.get('DualAttentionMode', 'None')
    self.NetDef_PreNorm = config_net_def.get('PreNorm', False)
    self.NetDef_NormType = config_net_def.get('NormType', 'LayerNorm')
    self.NetDef_AttentionMultiplier = config_net_def.get('AttentionMultiplier', 1)
    self.NetDef_NonLinearAttention = config_net_def.get('NonLinearAttention', False)
    self.NetDef_FFNMultiplier = config_net_def.get('FFNMultiplier', 1)
    self.NetDef_FFNActivationType = config_net_def.get('FFNActivationType', 'ReLUSquared')
    self.NetDef_HeadsActivationType = config_net_def.get('HeadsActivationType', 'ReLU')
    self.NetDef_PriorStateDim = config_net_def.get('PriorStateDim', 64)  
    self.NetDef_DeepNorm = config_net_def.get('DeepNorm', False) 
    self.NetDef_DenseFormer = config_net_def.get('DenseFormer', False) 
    self.NetDef_SmolgenDimPerSquare = config_net_def.get('SmolgenDimPerSquare', 32)
    self.NetDef_SmolgenDim = config_net_def.get('SmolgenDim', 512)
    self.NetDef_SmolgenToHeadDivisor = config_net_def.get('SmolgenToHeadDivisor', 2)
    self.NetDef_SmolgenActivationType = config_net_def.get('SmolgenActivationType', 'None')
    self.NetDef_HeadWidthMultiplier = config_net_def.get('HeadWidthMultiplier', 4)
    self.NetDef_UseRPE = config_net_def.get('UseRPE', False)
    self.NetDef_UseRelBias = config_net_def.get('UseRelBias', False)

    self.NetDef_TestValue = config_net_def.get('TestValue', 0)

    # SoftMoEConfig is a nested structure, so it requires special handling
    soft_moe_config = config_net_def.get('SoftMoEConfig', {})
    self.NetDef_SoftMoE_MoEMode = soft_moe_config.get('MoEMode', 'None')
    self.NetDef_SoftMoE_OnlyForAlternatingLayers = soft_moe_config.get('OnlyForAlternatingLayers', True)
    self.NetDef_SoftMoE_NumExperts = soft_moe_config.get('NumExperts', 0)
    self.NetDef_SoftMoE_NumSlotsPerExpert = soft_moe_config.get('NumSlotsPerExpert', 0)
    self.NetDef_SoftMoE_UseNormalization = soft_moe_config.get('UseNormalization', False)
    self.NetDef_SoftMoE_UseBias = soft_moe_config.get('UseBias', True)


  def pretty_print(self):
    print("")
    print("CONFIGURATION OF TRAINING RUN:", self.id)
    attributes = vars(self)
    sorted_attributes = sorted(attributes.items(), key=lambda x: x[0])

    # Pretty print the sorted attributes
    for attr_name, attr_value in sorted_attributes:
      print(f"{attr_name}: {attr_value}")


def Test():
  dir = '.'
  config = Configuration(dir, 'test')

  config.pretty_print()
