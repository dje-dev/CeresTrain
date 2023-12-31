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
import sys
import socket
import datetime
import numpy as np
from typing import Dict, Any

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
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
torch.backends.cuda.enable_flash_sdp(True)

#sys.argv = ['train.py', '/mnt/e/cout/configs/KRP_256_8_start35x', '/mnt/e/cout/nets']
#sys.argv = ['train.py', '/mnt/e/cout/configs/t1_wsl', '/mnt/e/cout/nets', 'CONVERT', '1525782528']

if len(sys.argv) < 3:
  raise ValueError("train.py expected <config_path> <save_nets_directory>")

TRAINING_ID = sys.argv[1]
SAVE_NET_DIR = sys.argv[2]
CONVERT_ONLY = len(sys.argv) >= 4 and sys.argv[3].upper() == 'CONVERT'

config = Configuration('.', TRAINING_ID)
TPG_TRAIN_DIR = config.Data_TrainingFilesDirectory 

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

NAME = socket.gethostname() + "_" + os.path.basename(TRAINING_ID)

accelerator = config.Exec_DeviceType.lower()
devices = config.Exec_DeviceIDs

BATCH_SIZE = config.Opt_BatchSizeBackwardPass

assert config.Opt_Optimizer == 'AdamW', 'only AdamW supported currently'
assert config.NetDef_PreNorm == False, 'PreNorm not supported'
assert config.Exec_DataType == 'BFloat16', 'Only BFloat16 training supported'

MAX_POSITIONS = config.Opt_NumTrainingPositions

LR = config.Opt_LearningRateBase
WEIGHT_DECAY = config.Opt_WeightDecay

NUM_TOKENS = 64 # N.B. ALSO UPDATE IN pt_net.py

num_pos = 0

time_last_status_update = datetime.datetime.now()
time_last_save = datetime.datetime.now()
time_start = datetime.datetime.now()
time_last_save_permanent = datetime.datetime.now()
time_last_save_transient = datetime.datetime.now()


def save_to_torchscript(fabric : Fabric, model : CeresNet, state : Dict[str, Any], net_step : str, save_onnx : str):
  CKPT_NAME = "ckpt_" + NAME + "_" + net_step

  # save PyTorch checkpoint
  fabric.save(os.path.join(SAVE_NET_DIR, CKPT_NAME), state)

  # save as TorchScript file
  SAVE_TS_NAME = CKPT_NAME + ".ts"
  SAVE_TS_PATH = os.path.join(SAVE_NET_DIR, SAVE_TS_NAME)
  print()
  print ('INFO: TORCHSCRIPT_FILENAME', SAVE_TS_NAME)

  # TODO: consider using lightning/fabric directly?  to_torchscript(file_path=None, method='script', example_inputs=None, **kwargs)
  with torch.no_grad():
    #model.load_state_dict(saved["model"])
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m.eval()
    sample_inputs = torch.rand(256, NUM_TOKENS, 135 * (64 // NUM_TOKENS)).to(torch.float32).to(m.device)
    if True:
      m_save = torch.jit.trace(m, sample_inputs)
    else:
      # NOTE: fails for some common Pytorch operations such as einops
      m_save = torch.jit.script(m)
    m_save.save(SAVE_TS_PATH)
    
    # below simpler method fails, probably due to use of .compile
    #BATCH_SIZE = 16
    #test_inputs_pytorch = torch.rand(BATCH_SIZE, NUM_TOKENS, 135 * (64 // NUM_TOKENS)).to(torch.float32).device(m.device)
    #model.to_torchscript(file_path=SAVE_PATH, method='trace', example_inputs=test_inputs_pytorch)
    #print('done save TS', SAVE_PATH )
    #model.to_onnx(SAVE_PATH + ".onnx", test_inputs_pytorch) #, export_params=True)

    try:
      ONNX_SAVE_PATH = SAVE_TS_PATH + ".onnx"
      ONNX16_SAVE_PATH = SAVE_TS_PATH + "_16.onnx"
      
      # still in beta testing as of PyTorch 2.1, not yet functional: torch.onnx.dynamo_export
      if save_onnx: # possibly hangs or crashes, possibly on in certain comiled modes
        torch.onnx.export(m,
                    [sample_inputs],
                    ONNX_SAVE_PATH,
                    do_constant_folding=True,
                    export_params=True,
                    opset_version=17,
                    input_names = ['squares'], 
                    output_names = ['policy', 'value', 'mlh', 'inc'], 
                    dynamic_axes={'squares' : {0 : 'batch_size'},    
                                  'policy' : {0 : 'batch_size'},
                                  'value' : {0 : 'batch_size'},
                                  'mlh' : {0 : 'batch_size'},
                                  'unc' : {0 : 'batch_size'}})
        print('INFO: ONNX_FILENAME', ONNX_SAVE_PATH)

        if True:
          from onnxmltools.utils.float16_converter import convert_float_to_float16
          from onnxmltools.utils import load_model, save_model
          onnx_model = load_model(ONNX_SAVE_PATH)
          onnx_model_16 = convert_float_to_float16(onnx_model)
          save_model(onnx_model_16, ONNX16_SAVE_PATH)
          print ('INFO: ONNX16_FILENAME', ONNX16_SAVE_PATH)

    except Exception as e:
      print(f"Warning: ONNX save failed, skipping. Exception details: {e}")
     
    model.train()


def Train():
  global num_pos
  global fraction_complete

  print("**** STARTING ", NAME)

  fabric = Fabric(precision="bf16-mixed", accelerator=accelerator, devices=devices,
                  loggers=TensorBoardLogger("./logs", name=NAME))  


  # NOTE: these very small values for MLH and UNC are best because
  #       they enhance training stability and don't negatively affect policy/value
  #       but produce MLH/UNC outputs which are not significantly less accurate 
  #       than if were at higher loss weight.
  model = CeresNet(fabric, config, policy_loss_weight=config.Opt_LossPolicyMultiplier,
                   value_loss_weight= config.Opt_LossValueMultiplier, 
                   moves_left_loss_weight= config.Opt_LossMLHMultiplier, 
                   unc_loss_weight= config.Opt_LossUNCMultiplier,
                   q_ratio=config.Data_FractionQ, optimizer='adamw', learning_rate=LR)

  if False:
    # Load the ONNX model
    import onnxruntime as ort
    ort_session = ort.InferenceSession("test.onnx")

    # Read the CSV file and reshape the data into the required input shape
    csv_file = '/mnt/superd/temp/input_data.csv'
    data = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.extend([float(x) for x in row])
    input_data = np.array(data, dtype=np.float32).reshape(1, 64, 20)

    # Run inference on the input data
    input_name = ort_session.get_inputs()[0].name
    output_names = [o.name for o in ort_session.get_outputs()]
    outputs = ort_session.run(output_names, {input_name: input_data})

    # Unpack the outputs
    policy, value, mlh, inc = outputs

    # Print outputs
    print("Policy: ", policy)
    print("Value: ", value)
    print("Mlh: ", mlh)
    print("Inc: ", inc)

  
  # N.B. when debugging, may be helpful to disable this line (otherwise breakpoints relating to graph evaluation will not be hit).
  if config.Opt_PyTorchCompileMode is not None:
    model = torch.compile(model, mode=config.Opt_PyTorchCompileMode, dynamic=False)  # choices:default, reduce-overhead, max-autotune 

  if CONVERT_ONLY:
    if len(sys.argv) < 5:
      raise("Missing required argument indicating step count of checkpoint file. Arguments expected: <config_id> <out_path> CONVERT <step>")
    net_step = sys.argv[4]
    save_to_torchscript(fabric, model, state, net_step, True)
    exit(0)

    onnx_model = model.to_onnx(EXPORT_NAME + ".onnx", 
      input_sample=[torch.zeros(1024, 64, 39 + (3 * 13)).to("cuda").to(torch.float32)], # TODO: 39 hardcoded
      export_params=True,
      opset_version=17,
      input_names = ['squares'], 
      output_names = ['policy', 'value', 'mlh', 'inc'], 
      dynamic_axes={'squares' : {0 : 'batch_size'},    
                    'policy' : {0 : 'batch_size'},
                    'value' : {0 : 'batch_size'},
                    'mlh' : {0 : 'batch_size'},
                    'unc' : {0 : 'batch_size'}})
    # Export FP16 version
    from onnxmltools.utils.float16_converter import convert_float_to_float16
    from onnxmltools.utils import load_model, save_model
    onnx_model = load_model(EXPORT_NAME + '.onnx')
    onnx_model_16 = convert_float_to_float16(onnx_model)
    save_model(onnx_model_16, EXPORT_NAME + '_16.onnx')

    exit()

  if False:
    DEVICE = 0
    model = model.cuda(DEVICE).half().eval()
    import time
    input = torch.randn(256, 64, 39 + (3 * 13)).cuda(DEVICE).half()   
    model.eval()
    for _ in range(20):
      x = model(input)
#      del x

    while True:
      with torch.no_grad():
        torch.cuda.synchronize(DEVICE)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(100 * 4):
          x = model(input)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)
        print(f"Elapsed time: {elapsed_time:.2f} ms")
         
        exit (3)


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
      {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": WEIGHT_DECAY},
      {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
  ]

  # Loss and optimizer
  optimizer = optim.AdamW(optim_groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=(config.Opt_Beta1, config.Opt_Beta2))

  fraction_complete = 0

  def lr_lambda(epoch : int):
    global fraction_complete

    FRAC_START_DELAY = config.Opt_LRBeginDecayAtFractionComplete
    FRAC_MIN = 0.10

    if fraction_complete < 0.02:
        lr_scale = 0.1 # warmup
    elif fraction_complete < FRAC_START_DELAY:
        lr_scale = 1.0
    else:
        # Linearly interpolate between 1.0 and FRAC_MIN as fracComplete goes from FRAC_START_DECAY to 1.0
        lr_scale = 1.0 - (fraction_complete - FRAC_START_DELAY) * (1.0 - FRAC_MIN) / (1.0 - FRAC_START_DELAY)

    return min(lr_scale, 1)

  scheduler = LambdaLR(optimizer, lr_lambda)

  state = {"model": model, "optimizer": optimizer, "num_pos" : num_pos}
 
  fabric.launch()
  model, optimizer = fabric.setup(model, optimizer)

  batch_size_forward = config.Opt_BatchSizeForwardPass

  def worker_init_fn(worker_id):
    dataset.set_worker_id(worker_id)

  NUM_DATASET_WORKERS = 2
  PREFETCH_FACTOR = 8 # to keep GPU busy

  world_size = len(devices)
  rank = 0 if world_size == 1 else dist.get_rank()
  dataset = TPGDataset(TPG_TRAIN_DIR, batch_size_forward // len(devices), config.Data_WDLLabelSmoothing, rank, world_size, NUM_DATASET_WORKERS)

  dataloader = DataLoader(dataset, batch_size=None, pin_memory=True, num_workers=NUM_DATASET_WORKERS, worker_init_fn=worker_init_fn, prefetch_factor=PREFETCH_FACTOR)
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

  # Train Network
  for batch_idx, (batch) in enumerate(dataloader):
    if (num_pos >= MAX_POSITIONS):
        break

    fraction_complete = num_pos / MAX_POSITIONS

    # Periodically log statistics
    show_losses =  (fabric.is_global_zero) and (num_pos % (1024 * 64) == 0)

    is_accumulating = ((batch_accumulation_counter + 1) % num_batches_gradient_accumulate) != 0
    with fabric.no_backward_sync(model, enabled=is_accumulating): # see https://lightning.ai/docs/fabric/stable/advanced/gradient_accumulation.html
      policy_out, value_out, moves_left_out, unc_out = model(batch['squares'])
      loss = model.compute_loss(loss_calc, batch, policy_out, value_out, moves_left_out, unc_out,
                                num_pos, scheduler.get_last_lr(), show_losses)

      fabric.backward(loss)

    if not is_accumulating:
      if config.Opt_GradientClipLevel > 0:
        fabric.clip_gradients(model, optimizer, max_norm=config.Opt_GradientClipLevel)
      scheduler.step()
      optimizer.step()
      optimizer.zero_grad()

    batch_accumulation_counter = batch_accumulation_counter + 1
    
    # update number of positions processed across all workers
    num_pos = num_pos + (fabric.world_size * batch['squares'].shape[0])


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

      STATUS_UPDATE_INTERVAL = 1
      should_show_status = (time_since_status_update > STATUS_UPDATE_INTERVAL) or (num_pos >= MAX_POSITIONS)

      should_save_permanent = time_since_save_permanent > 120 * 60 # permanent save named by step every 2 hours
      should_save_transient = time_since_save_transient > 30 * 60  # transient save as "last" every 30 minutes
         
      if should_save_permanent:
        save_to_torchscript(fabric, model, state, str(num_pos), True)
        time_last_save_permanent = datetime.datetime.now()

      if should_save_transient and not should_save_permanent:
        save_to_torchscript(fabric, model, state, "last", False)
        time_last_save_transient  = datetime.datetime.now()

      if should_show_status:
        # Note that this code executes only for primary worker (if multi-GPU),
        # and the statistics are collected over the recent training history only for that worker.
        # Although incomplete, the resulting statistics should nevertheless be reasonably accurate.
        total_loss =  (config.Opt_LossPolicyMultiplier * loss_calc.LAST_POLICY_LOSS
                     + config.Opt_LossValueMultiplier * loss_calc.LAST_VALUE_LOSS
                     + config.Opt_LossMLHMultiplier * loss_calc.LAST_MLH_LOSS
                     + config.Opt_LossUNCMultiplier * loss_calc.LAST_UNC_LOSS)
        
        print("TRAIN:", num_pos, ",", total_loss, ",", loss_calc.LAST_VALUE_LOSS, ",", loss_calc.LAST_POLICY_LOSS, ",", loss_calc.LAST_VALUE_ACC, ",", loss_calc.LAST_POLICY_ACC, ",", 
               loss_calc.LAST_MLH_LOSS, ",",  loss_calc.LAST_UNC_LOSS, ",", 
               scheduler.get_last_lr()[0], flush=True)
        loss_calc.reset_counters()
        time_last_status_update =  datetime.datetime.now()

  # final save and convert to Torchscript
  if (fabric.is_global_zero):
    save_to_torchscript(fabric, model, state, "final", True)

    # Emit special phrase to indicate end of training.
    print("INFO: EXIT_STATUS", "SUCCESS")

Train()


