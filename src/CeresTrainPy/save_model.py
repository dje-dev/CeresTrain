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
from typing import Dict, Any

import torch

from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger

from config import Configuration


def save_model(NAME : str, 
               OUTPUTS_DIR : str,
               config : Configuration,  
               fabric : Fabric, 
               model_nocompile,
               state : Dict[str, Any], 
               net_step : str, 
               save_all_formats : str):
  CKPT_NAME = "ckpt_" + NAME + "_" + net_step

  with torch.no_grad():
    convert_type = model_nocompile.dtype

    model_nocompile.eval()

    # AOT export. Works (generates .so file), but seemingly slower than ONNX export options.
    if False and fabric.is_global_zero and CONVERT_ONLY:
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
        aot_example_inputs = (torch.rand(256, 64, 137).to(convert_type).to(model_nocompile.device), 
                              torch.rand(256, 64, 4).to(convert_type).to(model_nocompile.device))
        with torch.no_grad():
          so_path = torch._export.aot_compile(model_nocompile,
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
    sample_inputs = [torch.rand(256, 64, 137).to(convert_type).to(model_nocompile.device), 
                     torch.rand(256, 64, config.NetDef_PriorStateDim).to(convert_type).to(model_nocompile.device)]

    if False and fabric.is_global_zero: # equivalent to below (this is just the raw PyTorch way rather than Lightning way above)
      try:
        SAVE_TS_PATH = os.path.join(OUTPUTS_DIR, 'nets', CKPT_NAME + "_jit.ts")
        m_save = torch.jit.trace(model_nocompile, sample_inputs)        
        #m_save = torch.jit.script(m) # NOTE: fails for some common Pytorch operations such as einops
        m_save.save(SAVE_TS_PATH)
        print('INFO: TS_JIT_FILENAME', SAVE_TS_PATH)
      except Exception as e:
        print(f"Warning: torchscript save failed, skipping. Exception details: {e}")
    
    SAVE_TS_PATH = os.path.join(OUTPUTS_DIR, 'nets', CKPT_NAME + ".ts")
    if fabric.is_global_zero:
      try:
        model_nocompile.to_torchscript(file_path=SAVE_TS_PATH, method='trace', example_inputs=sample_inputs)
        print('INFO: TS_FILENAME', SAVE_TS_PATH )
        #model.to_onnx(SAVE_PATH + ".onnx", test_inputs_pytorch) #, export_params=True)
      except Exception as e:
        print(f"Warning: to_torchscript save failed, skipping. Exception details: {e}")
    
    if save_all_formats:
      # Still in beta testing as of PyTorch 2.3, not yet functional: torch.onnx.dynamo_export
      # TorchDynamo based export. Encountered warning/error on export.
      if False and fabric.is_global_zero:
        ONNX_SAVE_PATH = SAVE_TS_PATH + ".dynamo.onnx"
        ONNX16_SAVE_PATH = SAVE_TS_PATH + "dynamo.fp16.onnx"

        try:
          export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
          onnx_model = torch.onnx.dynamo_export(model_nocompile, sample_inputs[0], sample_inputs[1], export_options=export_options)
          onnx_model.save(ONNX_SAVE_PATH)
          print('INFO: ONNX_DYNAMO_FILENAME', ONNX_SAVE_PATH)
        except Exception as e:
          print(f"Warning: torch.onnx.dynamo_export save failed, skipping. Exception details: {e}")

      # Legacy ONNX export.
      if True and fabric.is_global_zero:
        ONNX_SAVE_PATH = SAVE_TS_PATH + ".onnx"
        ONNX16_SAVE_PATH = SAVE_TS_PATH + ".fp16.onnx"

        try:
          head_output_names = ['policy', 'value', 'mlh', 'unc', 'value2', 
                               'q_deviation_lower', 'q_deviation_upper',
                               'uncertainty_policy', 'action', 'prior_state', 
                               'action_uncertainty']
          output_axes = {'squares' : {0 : 'batch_size'},    
                          'policy' : {0 : 'batch_size'},
                          'value' : {0 : 'batch_size'},
                          'mlh' : {0 : 'batch_size'},
                          'unc' : {0 : 'batch_size'},
                          'value2' : {0 : 'batch_size'},
                          'q_deviation_lower' : {0 : 'batch_size'},
                          'q_deviation_upper' : {0 : 'batch_size'},
                          'uncertainty_policy': {0 : 'batch_size'},
                          'action': {0 : 'batch_size'},
                          'prior_state': {0 : 'batch_size'},
                          'action_uncertainty': {0 : 'batch_size'},
                          }
          sample_inputs = (torch.rand(256, 64, 137).to(convert_type).to(model_nocompile.device), 
                            torch.rand(256, 64, config.NetDef_PriorStateDim).to(convert_type).to(model_nocompile.device))
          torch.onnx.export(model_nocompile,
                            (sample_inputs[0], sample_inputs[1]),
                            ONNX_SAVE_PATH,
                            do_constant_folding=True,
                            export_params=True,
                            opset_version=17, # Pytorch 2.3 maximum supported opset version 17
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
    # N.B. This should be called independent of fabric.is_global_zero (https://github.com/Lightning-AI/pytorch-lightning/issues/19780)    
    state_no_compile = {"model": model_nocompile, "optimizer": state['optimizer'], "num_pos" : net_step}
    fabric.save(os.path.join(OUTPUTS_DIR, 'nets', CKPT_NAME), state_no_compile)
    print ('INFO: CHECKPOINT_FILENAME', CKPT_NAME)
