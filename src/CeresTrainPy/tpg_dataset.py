# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

# NOTE: this code derived from: https://github.com/Rocketknight1/minimal_lczero.

import os, fnmatch
import numpy as np
import zstandard

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

# stable hash function for strings so all worker processes use same function
def stable_str_hash(s):
    return sum(ord(c) for c in s)

MAX_MOVES = 92 # Maximum number of policy moves in a position that can be stored (TPGRecord.MAX_MOVES)

class TPGDataset(Dataset):
  """
  TPGDataset is a subclass of the PyTorch Dataset for efficiently reading and parsing raw binary
  TPGRecords files (compressed in ZST format) containing training data for chess positions.
  It's optimized for high throughput (typically about 50,000 positions per second per worker) 
  and takes care to partition data without overlap across the possibly multiple workers in a distributed setup. 

  The class uses numpy for initial parsing of values (using ascontiguousarray, view, reshape) 
  and converts them into PyTorch tensors on the CPU. It's designed to work with a specific 
  binary format of chess training data (TPG).

  Parameters:
      root_dir (str): Directory containing TPGRecords files with ZST extension.
      batch_size (int): Size of the batch to be read and processed.
      wdl_smoothing (float): Smoothing parameter for win/draw/loss data.
      rank (int): The rank of the process in distributed training.
      world_size (int): Total number of processes in distributed training.
      num_workers (int): Number of worker processes for data loading.
  """
  def __init__(self, root_dir, batch_size, wdl_smoothing, rank, world_size, num_workers):

    self.root_dir = root_dir
    self.batch_size = batch_size
    self.wdl_smoothing = wdl_smoothing
    self.num_workers = num_workers
    self.generator = self.item_generator()
      
    # Get the list of files in the specified directory and select the subset appropriate for this worker.
    all_files = fnmatch.filter(os.listdir(root_dir), '*.zst')
    all_files.sort(key=lambda f: stable_str_hash(f))  # deterministic shuffle

    # Divide files as evenly as possible among workers
    files_per_worker = len(all_files) // world_size
    start_index = rank * files_per_worker
    end_index = start_index + files_per_worker

     # Assign files to this worker
    self.files = all_files[start_index:end_index]

    self.worker_id = None

    print('Creating TPGDataset at', root_dir, ' found', len(self.files), 'files matching this worker', rank, 'of', world_size, 'to be split among', num_workers, 'workers.')


  def set_worker_id(self, worker_id):
    self.worker_id = worker_id


  def __len__(self):
        # There is no actual limit (we repeat if necessary), so we just return a large number.
        # N.B. under some circumstances PyTorch will construct a data structure of this length, 
        #      so we return a number large enough to be more than any reasonable training session, but not excessive.
        return 10_000_000 # probably large enough (e.g. 10 million batches of size 1024 ==> 20 billion positions)


  def item_generator(self):
    DTYPE = np.float32
    BATCH_SIZE = self.batch_size
    BYTES_PER_POS = 9108 # fixed size of record structure (TPGRecord.TOTAL_BYTES)
    POS_PER_BLOCK = 24576//2 # read this many positions per loop iteration (somewhat arbitrary, each block about 115MB)
    BYTES_PER_BLOCK = POS_PER_BLOCK * BYTES_PER_POS

    if self.worker_id is None:
      print('ERROR: worker_id not initialized')
      exit()

    # Reduce files to be only the files that this worker is responsible for.
    assert self.worker_id >= 0, "Worker ID expected to have been be set before calling item_generator" 
    self.files = [file for index, file in enumerate(self.files) if index % self.num_workers == self.worker_id]      

    wdl_smoothing_transform = np.array([
        [1-self.wdl_smoothing, self.wdl_smoothing*0.6666, self.wdl_smoothing*0.3333],
        [self.wdl_smoothing*0.5, 1-self.wdl_smoothing, self.wdl_smoothing*0.5],
        [self.wdl_smoothing*0.3333, self.wdl_smoothing*0.6666, 1-self.wdl_smoothing]])

    while True:
      for file_name in self.files:
        print()
        print('DATASET WORKER', self.worker_id, 'PROCESSING TPG FILE', file_name)
        with open(os.path.join(self.root_dir, file_name),'rb') as file:
          dctx = zstandard.ZstdDecompressor()
          stream_reader = dctx.stream_reader(file)

          while True:
            decompressed_data = stream_reader.read(BYTES_PER_BLOCK)
            if (not decompressed_data) or len(decompressed_data) < BYTES_PER_BLOCK:
              break

            dd = np.frombuffer(decompressed_data, dtype=np.uint8)
            batches = dd.reshape(-1, BATCH_SIZE, BYTES_PER_POS)
            for batch_num in range(batches.shape[0]):
              this_batch = batches[batch_num,:,:]
              
              offset = 0 # running offset of where we are within the record

              # Read sequence of fields (see TPGRecord.cs)
              wdl_result = np.ascontiguousarray(this_batch[:, offset : offset + 3*4]).view(dtype=np.float32).reshape(-1, 3)
              if (self.wdl_smoothing > 0):
                wdl_result = np.matmul(wdl_result, wdl_smoothing_transform)

              if (self.wdl_smoothing == 0.5):
                assert 1==2, "wdl_smoothing == 0.5 not supported"

              offset+= 3 * 4

              wdl_q = np.ascontiguousarray(this_batch[:, offset : offset + 3*4]).view(dtype=np.float32).reshape(-1, 3)
              offset+= 3 * 4

              #ply_next_square_move = np.ascontiguousarray(this_batch[:, offset : offset + 64 * 1]).view(dtype=np.byte).reshape(-1, 64).astype(DTYPE)
              offset+= 64 * 1 # not currently used

              mlh = np.ascontiguousarray(this_batch[:, offset : offset + 1*4]).view(dtype=np.float32).reshape(-1, 1)
              mlh = np.square(mlh / 0.1) # undo preprocessing
              mlh = mlh / 100.
              offset+= 1 * 4

              uncertainty = np.ascontiguousarray(this_batch[:, offset : offset + 1*4]).view(dtype=np.float32).reshape(-1, 1)
              uncertainty = np.abs(uncertainty)
              offset+= 1 * 4

              offset+= 1 * 4 # unused

              policies_indices = np.ascontiguousarray(this_batch[:, offset : offset + MAX_MOVES*2]).view(dtype=np.int16).reshape(-1, MAX_MOVES)
              # much faster, but tries to reinitialize CUDA and fails:
              #   policies = torch.from_numpy(np.ascontiguousarray(this_batch[:, offset : offset + 1858*2]).view(dtype=np.float16)).cuda().reshape(-1,1858)
              offset+= MAX_MOVES * 2
              
              policies_values = np.ascontiguousarray(this_batch[:, offset : offset + MAX_MOVES*2]).view(dtype=np.float16).reshape(-1, MAX_MOVES)
              offset+= MAX_MOVES * 2

              SIZE_SQUARE = 135
              squares = np.ascontiguousarray(this_batch[:, offset : offset + 64 * SIZE_SQUARE * 1]).view(dtype=np.byte).reshape(-1, 64, SIZE_SQUARE).astype(DTYPE)
              DIVISOR = 100
              squares = np.divide(squares, DIVISOR).astype(DTYPE)
              offset+= 64 * SIZE_SQUARE

              assert(offset == BYTES_PER_POS)

              yield  ((policies_indices, policies_values, wdl_result, wdl_q, mlh, uncertainty, squares))


  def __getitem__(self, idx):
    batch = next(self.generator)
    policies_indices = batch[0]
    policies_values = batch[1]
    wdl_result = batch[2]
    wdl_q = batch[3]
    mlh = batch[4]
    uncertainty = batch[5]
    squares = batch[6]
    
    policies_indices = torch.tensor(policies_indices, dtype=torch.int64).reshape(self.batch_size, MAX_MOVES)
    policies_values  = torch.tensor(policies_values, dtype=torch.float16).reshape(self.batch_size, MAX_MOVES)

    # TO DO: do this on GPU?
    policies = torch.zeros(self.batch_size, 1858, dtype=torch.float16)
    policies.scatter_(1, policies_indices, policies_values)

    return {'policies': policies,
            'policies': policies, # NOT NEEDED
            'wdl_result':torch.tensor(wdl_result),
            'wdl_q': torch.tensor(wdl_q), 
            'mlh': torch.tensor(mlh), 
            'unc': torch.tensor(uncertainty), 
            'squares': torch.tensor(squares)}


def worker_init_fn(worker_id):
  """
    Initialize a worker function for a data loader.

    This function sets a global variable `WORKER_ID` to the ID of the worker.
    This method will be called in a multi-process data loading scenarios,
    allowing us to record the identifier if this worker for later coordination use.

    Args:
        worker_id (int): An integer identifier for the worker process.
    """
  global WORKER_ID
  WORKER_ID = worker_id



if __name__ == "__main__": 
  import time
  print('Beginning performance test of tpg_dataset.py.')
  TPG_TRAIN_DIR = "/mnt/e/scratch6" #"./test_data"
  devices = [0]
  BATCH_SIZE = 1024 * 4

  world_size = len(devices)
  rank = 0 if world_size == 1 else dist.get_rank()
  NUM_WORKERS = 2
  dataset = TPGDataset(TPG_TRAIN_DIR, BATCH_SIZE // len(devices), rank, world_size, NUM_WORKERS)
  tpg = DataLoader(dataset, batch_size=None, pin_memory=True, num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn, shuffle=False)

  BATCH_COUNT_PER_INTERVAL = 100
  start = time.time_ns()
  i = 0
  for batch_idx, (batch) in enumerate(tpg):
    if i % BATCH_COUNT_PER_INTERVAL == BATCH_COUNT_PER_INTERVAL - 1:
      end = time.time_ns()
      time_sec = (end-start)*0.001*0.001*0.001
      print (i, ' ', (BATCH_SIZE * BATCH_COUNT_PER_INTERVAL) / time_sec, '/sec')
      start = time.time_ns()
    i+=1

