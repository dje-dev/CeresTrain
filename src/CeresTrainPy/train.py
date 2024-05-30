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
import fnmatch
import sys
import socket
import datetime
import numpy as np
from typing import Dict, Any

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#from torchsummary import summary
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.distributed as dist

from rms_norm import RMSNorm
from losses import LossCalculator
from tpg_dataset import TPGDataset
from config import Configuration

from ceres_net import CeresNet
from soft_moe_batched_dual import SoftMoEBatchedDual
from multi_expert import MultiExpertLayer

from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger

print(torch.__version__)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True) # efficient seems faster than flash for short sequences


if len(sys.argv) < 3:
  raise ValueError("expected: train.py <config_name> <outputs_directory>")

TRAINING_ID = sys.argv[1]
OUTPUTS_DIR = sys.argv[2]

config = Configuration('.', os.path.join(OUTPUTS_DIR, "configs", TRAINING_ID))
TPG_TRAIN_DIR = config.Data_TrainingFilesDirectory 

if len(sys.argv) > 3 and sys.argv[3].upper() == 'CONVERT':
  #  "expected: train.py <config_name> <outputs_directory> CONVERT <path_ts_to_load>")
  # Settings to facilitate interactive debugging:
  # N.B. probably need to also:
  #  - set TPG_TRAIN_DIR to dummy value (possibly)
  #  - disable torch.compile
  #os.chdir('/home/david/dev/CeresTrain/src/CeresTrainPy')
  #sys.argv = ['train.py', '/mnt/deve/cout/configs/C5_B1_512_15_16_4_32bn_2024', '/mnt/deve/cout']
  CONVERT_ONLY = True
  config.Opt_PyTorchCompileMode = None
  config.Exec_DeviceIDs = [0] # always build based on GPU at index 0
  TPG_TRAIN_DIR = "." # path not actually used/needed
  if len(sys.argv) < 5:
    raise ValueError("expected: train.py <config_name> <outputs_directory> CONVERT <path_ts_to_load>")
  checkpoint_path_to_load = sys.argv[4]
  config.Opt_StartingCheckpointFN = os.path.join(OUTPUTS_DIR, 'nets', checkpoint_path_to_load)
else:
  CONVERT_ONLY = False

#  if CONVERT_ONLY:
#    if len(sys.argv) < 5:
#      raise("Missing required argument indicating step count of checkpoint file. Arguments expected: <config_id> <out_path> CONVERT <step>")
#    net_step = sys.argv[4]
#    save_to_torchscript(fabric, model, state, net_step, True)
#    exit(0)

#TODO: would be better to use asserts but they are not captured by the remote process executor
if TPG_TRAIN_DIR is None:
  print('ERROR: TrainingFilesDirectory is null')
  exit(1)
elif not os.path.isdir(TPG_TRAIN_DIR): 
  print('ERROR: TrainingFilesDirectory does not exist:', TPG_TRAIN_DIR)
  exit(1)
elif not os.listdir(TPG_TRAIN_DIR):
  print(f"ERROR: The directory TrainingFilesDirectory ('{TPG_TRAIN_DIR}') is empty.")
  exit(1)


def print_model_trainable_details(model):
  num_params = 0
  num_layers = 0
  print("Model details (trainable parameters only):\n")
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(f"Layer: {name} | Size: {param.size()} | Total parameters: {param.numel()}")
      num_params+= param.numel()
      num_layers = num_layers + 1
  print()
  print("INFO: NUM_PARAMETERS", str(num_params))

NAME = socket.gethostname() + "_" + TRAINING_ID

accelerator = config.Exec_DeviceType.lower()
devices = config.Exec_DeviceIDs

BATCH_SIZE = config.Opt_BatchSizeBackwardPass

assert config.NetDef_PreNorm == False, 'PreNorm not supported'
assert config.Exec_DataType == 'BFloat16', 'Only BFloat16 training supported'

MAX_POSITIONS = config.Opt_NumTrainingPositions

if config.Opt_LossActionMultiplier > 0 or config.NetDef_PriorStateDim > 0:
  BOARDS_PER_BATCH = 4
else:
  BOARDS_PER_BATCH = 1  

LR = config.Opt_LearningRateBase
WEIGHT_DECAY = config.Opt_WeightDecay

num_pos = 0

time_last_status_update = datetime.datetime.now()
time_last_save = datetime.datetime.now()
time_start = datetime.datetime.now()
time_last_save_permanent = datetime.datetime.now()
time_last_save_transient = datetime.datetime.now()


def save_to_torchscript(fabric : Fabric, model : CeresNet, state : Dict[str, Any], net_step : str, save_all_formats : str):
  CKPT_NAME = "ckpt_" + NAME + "_" + net_step
  #fabric.barrier() # try to prevent problems with hanging

  with torch.no_grad():
    convert_type = model.dtype  #torch.float16

    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m.eval()

    # AOT export. Works (generates .so file)
    if False and CONVERT_ONLY:
      try:
        #m = m.cuda().to(convert_type) # this might be necessary for AOT convert, but may cause subsequent failures if running net

        # get a device capabilities string (such as cuda_sm90)
        if torch.cuda.is_available():
          device = torch.cuda.get_device_properties(0)
          compute_capability = device.major, device.minor
          hardware_postfix = f"_cuda_sm{compute_capability[0]}{compute_capability[1]}" 
        else:
          hardware_postfix = "_cpu"

        #prepare output file name and directory
        aot_output_dir = "./" + TRAINING_ID
        aot_output_path = os.path.join(aot_output_dir, TRAINING_ID + hardware_postfix + ".so")
        if not os.path.exists(aot_output_dir):
          os.mkdir(aot_output_dir)
          
        batch_dim = torch.export.Dim("batch", min=1, max=1024)
        aot_example_inputs = (torch.rand(256, 64, 137).to(convert_type).to(m.device), 
                              torch.rand(256, 64, 4).to(convert_type).to(m.device))
        with torch.no_grad():
          so_path = torch._export.aot_compile(model,
                                            aot_example_inputs,
                                            dynamic_shapes={"squares": {0: batch_dim}, "prior_state": {0: batch_dim}},
                                            options={"aot_inductor.output_path": aot_output_path,
                                                    "max_autotune" : True,
#                                                    "max_autotune_gemm" : True,
#                                                    "max_autotune_pointwise" : True,
#                                                    "shape_padding" : True,
#                                                    "permute_fusion":True
                                                    })
        print('INFO: AOT_BINARY', so_path)
        exit(3)
      except Exception as e:
        print(f"Warning: torch._export.aot_compile save failed, skipping. Exception details: {e}")
  


    # below simpler method fails, probably due to use of .compile
    sample_inputs = [torch.rand(256, 64, 137).to(convert_type).to(m.device), 
                     torch.rand(256, 64, config.NetDef_PriorStateDim).to(convert_type).to(m.device)]
    if True:
      try:
        SAVE_TS_PATH = os.path.join(OUTPUTS_DIR, 'nets', CKPT_NAME + ".ts")
        m.to_torchscript(file_path=SAVE_TS_PATH, method='trace', example_inputs=sample_inputs)
        print('INFO: TS_FILENAME', SAVE_TS_PATH )
        #model.to_onnx(SAVE_PATH + ".onnx", test_inputs_pytorch) #, export_params=True)
      except Exception as e:
        print(f"Warning: to_torchscript save failed, skipping. Exception details: {e}")

    if False: # equivalent to above (this is just the raw PyTorch way rather than Lightning way above)
      try:
        SAVE_TS_PATH = os.path.join(OUTPUTS_DIR, 'nets', CKPT_NAME + "_jit.ts")
        m_save = torch.jit.trace(m, sample_inputs)        
        #m_save = torch.jit.script(m) # NOTE: fails for some common Pytorch operations such as einops
        m_save.save(SAVE_TS_PATH)
        print('INFO: TS_JIT_FILENAME', SAVE_TS_PATH )
      except Exception as e:
        print(f"Warning: torchscript save failed, skipping. Exception details: {e}")
    
    
    if save_all_formats:
      ONNX_SAVE_PATH = SAVE_TS_PATH + ".onnx"
      ONNX16_SAVE_PATH = SAVE_TS_PATH + ".fp16.onnx"

      # still in beta testing as of PyTorch 2.1, not yet functional: torch.onnx.dynamo_export
      # TorchDynamo based export. Works, but when try to do inference from C# it fails on second call (reshape)
      if False:
        try:
          export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
          onnx_model = torch.onnx.dynamo_export(m, sample_inputs[0], sample_inputs[1], export_options=export_options)
          onnx_model.save(ONNX_SAVE_PATH)
          print('INFO: ONNX_FILENAME', ONNX_SAVE_PATH)
        except Exception as e:
          print(f"Warning: torch.onnx.dynamo_export save failed, skipping. Exception details: {e}")

      # Legacy ONNX export.
      if True:
        try:
          head_output_names = ['policy', 'value', 'mlh', 'unc', 'value2', 'q_deviation_max', 'uncertainty_policy', 'action', 'prior_state', 'action_uncertainty']
          output_axes = {'squares' : {0 : 'batch_size'},    
                          'policy' : {0 : 'batch_size'},
                          'value' : {0 : 'batch_size'},
                          'mlh' : {0 : 'batch_size'},
                          'unc' : {0 : 'batch_size'},
                          'value2' : {0 : 'batch_size'},
                          'q_deviation_max' : {0 : 'batch_size'},
                          'uncertainty_policy': {0 : 'batch_size'},
                          'action': {0 : 'batch_size'},
                          'prior_state': {0 : 'batch_size'},
                          'action_uncertainty': {0 : 'batch_size'},
                          }
          sample_inputs = (torch.rand(256, 64, 137).to(convert_type).to(m.device), 
                            torch.rand(256, 64, config.NetDef_PriorStateDim).to(convert_type).to(m.device))
          torch.onnx.export(m,
                            (sample_inputs[0], sample_inputs[1]),
                            ONNX_SAVE_PATH,
                            do_constant_folding=True,
                            export_params=True,
                            opset_version=17,
                            input_names = ['squares', 'prior_state'], # if config.NetDef_PriorStateDim > 0 else ['squares'],
                            output_names = head_output_names, 
                            dynamic_axes=output_axes)
          print('INFO: ONNX_FILENAME', ONNX_SAVE_PATH)

          if True:
            # Make a 16 bit version
            from onnxmltools.utils.float16_converter import convert_float_to_float16
            from onnxmltools.utils import load_model, save_model
            onnx_model = load_model(ONNX_SAVE_PATH)
            onnx_model_16 = convert_float_to_float16(onnx_model)
            save_model(onnx_model_16, ONNX16_SAVE_PATH)
            print ('INFO: ONNX16_FILENAME', ONNX16_SAVE_PATH)

        except Exception as e:
          print(f"Warning: torch.onnx.export save failed, skipping. Exception details: {e}")       


    # Save PyTorch checkpoint.
    # N.B. If running multi-GPU, this tends to hang for unknown reasons.
    #      Therefore if multi-GPU do not checkpoint (unless triggered with special file)
    #if devices.count == 1 or os.path.isfile("FORCE_CHECKPOINT"): # or net_step == "final" 
    if False:  
      fabric.barrier() # try to prevent problems with hanging
      state_no_compile = {"model": m, "optimizer": state['optimizer'], "num_pos" : num_pos}
      fabric.save(os.path.join(OUTPUTS_DIR, 'nets', CKPT_NAME), state_no_compile)
      print ('INFO: CHECKPOINT_FILENAME', CKPT_NAME)


    model.train()


def Train():
  global num_pos
  global fraction_complete

  print("**** STARTING ", NAME)


  if config.Exec_UseFP8:
#    from lightning.fabric.plugins import TransformerEnginePrecision
#    recipe = {"fp8_format": "HYBRID", "amax_history_len": 16, "amax_compute_algo": "max"}
#    precision = TransformerEnginePrecision(dtype=torch.bfloat16, recipe=recipe, replace_layers=False)
#    fabric = Fabric(plugins=precision,accelerator=accelerator, devices=devices,
#                    loggers=TensorBoardLogger(os.path.join(OUTPUTS_DIR, 'tblogs'), name=NAME))  
    fabric = Fabric(precision="transformer-engine",accelerator=accelerator, devices=devices,
                    loggers=TensorBoardLogger(os.path.join(OUTPUTS_DIR, 'tblogs'), name=NAME))  
  else:
    fabric = Fabric(precision="bf16-mixed", accelerator=accelerator, devices=devices,
                    loggers=TensorBoardLogger(os.path.join(OUTPUTS_DIR, 'tblogs'), name=NAME))  


  # NOTE: these very small values for MLH and UNC are best because
  #       they enhance training stability and don't negatively affect policy/value
  #       but produce MLH/UNC outputs which are not significantly less accurate 
  #       than if were at higher loss weight.
  model = CeresNet(fabric, config, policy_loss_weight=config.Opt_LossPolicyMultiplier,
                   value_loss_weight= config.Opt_LossValueMultiplier, 
                   moves_left_loss_weight= config.Opt_LossMLHMultiplier, 
                   unc_loss_weight= config.Opt_LossUNCMultiplier,
                   value2_loss_weight= config.Opt_LossValue2Multiplier,
                   q_deviation_loss_weight= config.Opt_LossQDeviationMultiplier,
                   value_diff_loss_weight = config.Opt_LossValueDMultiplier,
                   value2_diff_loss_weight = config.Opt_LossValue2DMultiplier,
                   action_loss_weight = config.Opt_LossActionMultiplier,
                   uncertainty_policy_weight = config.Opt_LossUncertaintyPolicyMultiplier,
                   action_uncertainty_loss_weight = config.Opt_LossActionUncertaintyMultiplier,
                   q_ratio=config.Data_FractionQ)

   
  # N.B. when debugging, may be helpful to disable this line (otherwise breakpoints relating to graph evaluation will not be hit).
  if config.Opt_PyTorchCompileMode is not None:
    model = torch.compile(model, mode=config.Opt_PyTorchCompileMode, dynamic=False)  # choices:default, reduce-overhead, max-autotune 

  
  # carefully set weight decay to apply only to appropriate subset of parameters
  # based on code from: https://github.com/karpathy/minGPT
  whitelist_weight_modules = (torch.nn.Linear, SoftMoEBatchedDual, MultiExpertLayer)
  blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm)

  decay = set()
  no_decay = set()

  for mn, m in model.named_modules():
      for pn, p in m.named_parameters():
          fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
          if pn.endswith('bias'):
              no_decay.add(fpn)
          elif "rpe" in fpn:
              decay.add(fpn)
          elif "rpe_factor" in fpn:
              pass
          elif "alphas" in fpn: # for Denseformer
              decay.add(fpn)
          elif ".mem_" in fpn:
              decay.add(fpn)
          elif "qkv" in fpn:
              decay.add(fpn)
          elif "embedding" in fpn:
              no_decay.add(fpn)
          elif isinstance(m, blacklist_weight_modules):
              no_decay.add(fpn)
          elif isinstance(m, whitelist_weight_modules):
              decay.add(fpn)

  
  param_dict = {pn: p for pn, p in model.named_parameters()}
  inter_params = decay & no_decay
  union_params = decay | no_decay
  assert len(inter_params) == 0, "parameters %s appear in both decay/no_decay sets" % (str(inter_params), )
  assert len(param_dict.keys() - union_params) == 0, "parameters %s were not fully partitioned into decay/no_decay sets" \
                                              % (str(param_dict.keys() - union_params), ) 
        
  optim_groups = [
      {"params": [param_dict[pn] for pn in sorted(list(decay))  if "rpe_factor" not in pn], "weight_decay": WEIGHT_DECAY},
      {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if "rpe_factor" not in pn], "weight_decay": 0.0},
  ]

  # Loss and optimizer
  if config.Opt_Optimizer == 'NAdamW':
    optimizer = optim.NAdam(optim_groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=(config.Opt_Beta1, config.Opt_Beta2), decoupled_weight_decay=True)
  elif config.Opt_Optimizer == 'AdamW':
    optimizer = optim.AdamW(optim_groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=(config.Opt_Beta1, config.Opt_Beta2), fused=False)
  else:
    raise ValueError("Unsupported optimizer: " + config.Opt_Optimizer)

  fraction_complete = 0

  def lr_lambda(epoch : int):
    global fraction_complete
    global num_pos
    
    FRAC_START_DELAY = config.Opt_LRBeginDecayAtFractionComplete
    FRAC_MIN = 0.15

    if num_pos < 20000000 and (fraction_complete < 0.02 or num_pos < 500000):
      return FRAC_MIN # warmup
    elif fraction_complete < FRAC_START_DELAY:
      return 1.0
    elif fraction_complete > 1:
      return FRAC_MIN # shouldn't happen
    else:
      # Once decay starts, LR multiplier starts at fraction remaining and linearly decreases to FRAC_MIN
      fraction_remaining = 1.0 - fraction_complete
      frac_end_delay = 1.0 - FRAC_START_DELAY
      return FRAC_MIN + (fraction_remaining/frac_end_delay) * (frac_end_delay - FRAC_MIN)      

  scheduler = LambdaLR(optimizer, lr_lambda)

  state = {"model": model, "optimizer": optimizer, "num_pos" : num_pos}


  if False:
    torchscript_model = torch.jit.load("/mnt/deve/cout/nets/ckpt_DGX_C5_B1_512_15_16_4_32bn_2024_final_jit.ts")
    with torch.no_grad():
      for pytorch_param, torchscript_param in zip(model.parameters(), torchscript_model.parameters()):
         pytorch_param.data.copy_(torchscript_param.data)
    del torchscript_model

  
  # Sample code to load from a saved TorchScript model (and possibly save back)
  if CONVERT_ONLY:
    if config.Opt_StartingCheckpointFN is None:   
      raise ValueError("No starting checkpoint specified for conversion")
    else:
      print("loading ", config.Opt_StartingCheckpointFN)
      torchscript_model = torch.jit.load(config.Opt_StartingCheckpointFN)
      with torch.no_grad():
        for pytorch_param, torchscript_param in zip(model.parameters(), torchscript_model.parameters()):
            pytorch_param.data.copy_(torchscript_param.data)
      del torchscript_model
    
      print("converting....")
      save_to_torchscript(fabric, model, state, "fix", True)
      exit(3)      
 
  fabric.launch()
  model, optimizer = fabric.setup(model, optimizer)

  batch_size_forward = config.Opt_BatchSizeForwardPass

  def worker_init_fn(worker_id):
    dataset.set_worker_id(worker_id)

  # Use two concurrent dataset workers (if more than one training data file is available)
  count_zst_files = len(fnmatch.filter(os.listdir(TPG_TRAIN_DIR), '*.zst'))
  NUM_DATASET_WORKERS = min(1, count_zst_files)
  PREFETCH_FACTOR = 4 # to keep GPU busy

 
  world_size = len(devices)
  rank = 0 if world_size == 1 else dist.get_rank()
  dataset = TPGDataset(TPG_TRAIN_DIR, batch_size_forward // len(devices), config.Data_WDLLabelSmoothing, rank, world_size, NUM_DATASET_WORKERS, BOARDS_PER_BATCH)

  dataloader = DataLoader(dataset, batch_size=None, pin_memory=False, num_workers=NUM_DATASET_WORKERS, worker_init_fn=worker_init_fn, prefetch_factor=PREFETCH_FACTOR)
  dataloader = fabric.setup_dataloaders(dataloader)

  if rank == 0:
    config.pretty_print()
    print_model_trainable_details(model)


  NUM_POS_TO_SKIP = 0

  if config.Opt_StartingCheckpointFN is not None:
    loaded = fabric.load(config.Opt_StartingCheckpointFN)
    model.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])
    num_pos = config.Opt_StartingCheckpointLastPosNum # N.B. be sure to use a multiple of the batch size
    # NUM_POS_TO_SKIP = num_pos # enable this line if want to skip training data already seen (but slow)
    del loaded

  # compute batch sizes
  batch_size_opt = config.Opt_BatchSizeBackwardPass
  assert batch_size_opt >= batch_size_forward and batch_size_opt % batch_size_forward == 0, 'data batch size must be be multiple of optimization batch size'
  num_batches_gradient_accumulate = batch_size_opt // batch_size_forward
  batch_accumulation_counter = 0

  loss_calc = LossCalculator()

  model.train()

  wdl_reverse = torch.tensor([2, 1, 0]) # for reversing perspective on WDL
 
  # Train Network
  for batch_idx, (batch) in enumerate(dataloader):
    if (num_pos >= MAX_POSITIONS):
        break

    fraction_complete = num_pos / MAX_POSITIONS

    # Periodically log statistics
    show_losses =  (fabric.is_global_zero) and (num_pos % (1024 * 64) == 0)

    is_accumulating = ((batch_accumulation_counter + 1) % num_batches_gradient_accumulate) != 0
    with fabric.no_backward_sync(model, enabled=is_accumulating): # see https://lightning.ai/docs/fabric/stable/advanced/gradient_accumulation.html
      this_lr = scheduler.get_last_lr()
      
      if BOARDS_PER_BATCH == 1:
        batch = batch[0]
        num_processing_now = batch['squares'].shape[0]
        policy_out, value_out, moves_left_out, unc_out, value2_out, q_deviation_max, uncertainty_policy_out, _, _, _ = model(batch['squares'], None)
        loss = model.compute_loss(loss_calc, batch, policy_out, value_out, moves_left_out, unc_out,
                                  value2_out, q_deviation_max, uncertainty_policy_out, 
                                  None, None, 
                                  None, None,
                                  None,
                                  0, num_pos, this_lr, show_losses)

      else:
        assert BOARDS_PER_BATCH == 4

        # Weights for the action loss terms.
        # The training data has 2 positions which are always optimal (or nearly optimal) moves
        # for every 1 which more evenly distributed over possible moves (of all quality).
        # To compensate for this non-representative training data distribution,
        # we give less weight to the over-sampled best continuation moves.
        LOSS_WEIGHT_ACTION_BEST_CONTINUATION = 0.15
        LOSS_WEIGHT_ACTION_RANDOM_CONTINUATION = 1.0
        
        # Note the logic below is hardcoded to use value, not value2.
        ACTION_HEAD_USES_PRIMARY_VALUE = True
        
        num_processing_now = batch[0]['squares'].shape[0] * BOARDS_PER_BATCH
        
        #Board 1
        sub_batch = batch[0]
        policy_out1, value_out1, moves_left_out1, unc_out1, value2_out1, q_deviation_max1, uncertainty_policy_out1, action_out1, state_out1, action_uncertainty_out1 = model(sub_batch['squares'], None)
        loss1 = model.compute_loss(loss_calc, sub_batch, policy_out1, value_out1, moves_left_out1, unc_out1,
                                   value2_out1, q_deviation_max1, uncertainty_policy_out1, 

                                   None, None, 
                                   None, None, 
                                   action_uncertainty_out1,
                                   
                                   0, num_pos, this_lr, show_losses)
        
        # Board 2
        sub_batch = batch[1]
        policy_out2, value_out2, moves_left_out2, unc_out2, value2_out2, q_deviation_max2, uncertainty_policy_out2, action_out2, state_out2, action_uncertainty_out2 = model(sub_batch['squares'], state_out1)

        if config.Opt_LossActionMultiplier > 0:
          action2_played_move_indices = sub_batch['policy_index_in_parent'].to(dtype=torch.int)
          extracted_action1_out = action_out1[torch.arange(0, action_out1.size(0)), action2_played_move_indices.squeeze(-1)]
          extracted_action1_out = extracted_action1_out[:, wdl_reverse]
        else:
          extracted_action1_out = None
          
        loss2 = model.compute_loss(loss_calc, sub_batch, policy_out2, value_out2, moves_left_out2, unc_out2,
                                   value2_out2, q_deviation_max2, uncertainty_policy_out2, 

                                   value_out1[:, wdl_reverse], value2_out1[:, wdl_reverse], # prior value outputs for value differencing
                                   value2_out2.detach(), extracted_action1_out,  # action target/output from previous board
                                   action_uncertainty_out2,
                                   
                                   LOSS_WEIGHT_ACTION_BEST_CONTINUATION, num_pos, this_lr, show_losses)
        
        # Board 3
        sub_batch = batch[2]
        policy_out3, value_out3, moves_left_out3, unc_out3, value2_out3, q_deviation_max3, uncertainty_policy_out3, action_out3, _, action_uncertainty_out3 = model(sub_batch['squares'], state_out2)

        if config.Opt_LossActionMultiplier > 0:
          action3_played_move_indices = sub_batch['policy_index_in_parent'].to(dtype=torch.int)
          extracted_action2_out = action_out2[torch.arange(0, action_out2.size(0)), action3_played_move_indices.squeeze(-1)]
          extracted_action2_out = extracted_action2_out[:, wdl_reverse]
        else:
          extracted_action2_out = None

        loss3 = model.compute_loss(loss_calc, sub_batch, policy_out3, value_out3, moves_left_out3, unc_out3,
                                   value2_out3, q_deviation_max3, uncertainty_policy_out3,

                                   value_out2[:, wdl_reverse], value2_out2[:, wdl_reverse], # prior value outputs for value differencing
                                   value2_out3.detach(), extracted_action2_out, # action target/output from previous board
                                   action_uncertainty_out3,

                                   LOSS_WEIGHT_ACTION_BEST_CONTINUATION, num_pos, this_lr, show_losses)

        # Board 4 (only used if action loss is enabled)
        if config.Opt_LossActionMultiplier > 0:
          sub_batch = batch[3]
          policy_out4, value_out4, moves_left_out4, unc_out4, value2_out4, q_deviation_max4, uncertainty_policy_out4, action_out4, _, action_uncertainty_out4 = model(sub_batch['squares'], None)


          action4_played_move_indices = sub_batch['policy_index_in_parent'].to(dtype=torch.int)
          extracted_action1_other_out = action_out1[torch.arange(0, action_out1.size(0)), action4_played_move_indices.squeeze(-1)]
          extracted_action1_other_out = extracted_action1_other_out[:, wdl_reverse]
          
          loss4 = model.compute_loss(loss_calc, sub_batch, None, None, None, None,
                                     None, None, None, 

                                     None, None,
                                     value2_out4.detach(), extracted_action1_other_out, # action target/output from previous board
                                     action_uncertainty_out4,
                                     
                                     LOSS_WEIGHT_ACTION_RANDOM_CONTINUATION, num_pos, this_lr, show_losses)

        if config.Opt_LossActionMultiplier > 0:
          loss = (loss1 + loss2 + loss3 + loss4) / 3 # although there are 4 loss terms, the last one is typically very small so we only divide by 3
        else:          
          loss = (loss1 + loss2 + loss3) / 3 # only 3 boards used

      fabric.backward(loss)
        
    if not is_accumulating:
      if config.Opt_GradientClipLevel > 0:
        fabric.clip_gradients(model, optimizer, max_norm=config.Opt_GradientClipLevel)
      scheduler.step()
      optimizer.step()
      optimizer.zero_grad()

    batch_accumulation_counter = batch_accumulation_counter + 1
    
    # update number of positions processed across all workers
    num_pos = num_pos + (fabric.world_size * num_processing_now)


    if (fabric.is_global_zero):
      current_time = datetime.datetime.now()

      global time_start
      global time_last_status_update
      global time_last_save_transient
      global time_last_save_permanent

      time_since_start = (current_time - time_start).seconds
      time_since_status_update = (current_time - time_last_status_update).seconds
      time_since_save_transient = (current_time - time_last_save_transient).seconds
      time_since_save_permanent = (current_time - time_last_save_permanent).seconds

      STATUS_UPDATE_INTERVAL = 5
      should_show_status = (time_since_status_update > STATUS_UPDATE_INTERVAL) or (num_pos >= MAX_POSITIONS)

      should_save_permanent = time_since_save_permanent > 120 * 60 # permanent save named by step every 2 hours
      should_save_transient = time_since_save_transient > 45 * 60  # transient save as "last" every 45 minutes
         
      if should_save_permanent:
        save_to_torchscript(fabric, model, state, str(num_pos), True)
        time_last_save_permanent = datetime.datetime.now()

      if should_save_transient and not should_save_permanent:
        save_to_torchscript(fabric, model, state, "last", True)
        time_last_save_transient  = datetime.datetime.now()

      if should_show_status:
        # Note that this code executes only for primary worker (if multi-GPU),
        # and the statistics are collected over the recent training history only for that worker.
        # Although incomplete, the resulting statistics should nevertheless be reasonably accurate.
        total_loss =  (config.Opt_LossPolicyMultiplier * loss_calc.LAST_POLICY_LOSS
                     + config.Opt_LossValueMultiplier * loss_calc.LAST_VALUE_LOSS
                     + config.Opt_LossValue2Multiplier * loss_calc.LAST_VALUE2_LOSS
                     + config.Opt_LossMLHMultiplier * loss_calc.LAST_MLH_LOSS
                     + config.Opt_LossUNCMultiplier * loss_calc.LAST_UNC_LOSS
                     + config.Opt_LossQDeviationMultiplier * loss_calc.LAST_Q_DEVIATION_MAX_LOSS       
                     + config.Opt_LossUncertaintyPolicyMultiplier * loss_calc.LAST_UNCERTAINTY_POLICY_LOSS
                     
                     + config.Opt_LossValueDMultiplier * loss_calc.LAST_VALUE_DIFF_LOSS
                     + config.Opt_LossValue2DMultiplier * loss_calc.LAST_VALUE2_DIFF_LOSS
                     
                     + config.Opt_LossActionMultiplier * loss_calc.LAST_ACTION_LOSS)

        
        # Note that this output line is parsed by the C# class CeresTrainProgressLoggingLine
        print("TRAIN:", num_pos, ",", 
              total_loss, ",", 
              loss_calc.LAST_VALUE_LOSS if config.Opt_LossValueMultiplier > 0 else 0, ",", 
              loss_calc.LAST_POLICY_LOSS if config.Opt_LossPolicyMultiplier > 0 else 0, ",", 
              loss_calc.LAST_VALUE_ACC if config.Opt_LossValueMultiplier > 0 else 0, ",", 
              loss_calc.LAST_POLICY_ACC if config.Opt_LossPolicyMultiplier > 0 else 0, ",", 
              loss_calc.LAST_MLH_LOSS if config.Opt_LossMLHMultiplier > 0 else 0, ",",  
              loss_calc.LAST_UNC_LOSS if config.Opt_LossUNCMultiplier > 0 else 0, ",", 
              loss_calc.LAST_VALUE2_LOSS if config.Opt_LossValue2Multiplier > 0 else 0, ",", 
              loss_calc.LAST_Q_DEVIATION_MAX_LOSS if config.Opt_LossQDeviationMultiplier > 0 else 0, ",", 
              loss_calc.LAST_UNCERTAINTY_POLICY_LOSS if config.Opt_LossUncertaintyPolicyMultiplier > 0 else 0, ",", 

              loss_calc.LAST_VALUE_DIFF_LOSS if config.Opt_LossValueDMultiplier > 0 else 0, ",", 
              loss_calc.LAST_VALUE2_DIFF_LOSS if config.Opt_LossValue2DMultiplier > 0 else 0, ",", 

              loss_calc.LAST_ACTION_LOSS if config.Opt_LossActionMultiplier > 0 else 0, ",",
              loss_calc.LAST_ACTION_UNCERTAINTY_LOSS if config.Opt_LossActionUncertaintyMultiplier > 0 else 0, ",",
              
              scheduler.get_last_lr()[0], flush=True)
        loss_calc.reset_counters()
        time_last_status_update = datetime.datetime.now()

  # final save and convert to Torchscript
  if (fabric.is_global_zero):
    save_to_torchscript(fabric, model, state, "final", True)

    # Emit special phrase to indicate end of training.
    print("INFO: EXIT_STATUS", "SUCCESS")

Train()


