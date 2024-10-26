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
def stable_str_hash(s: str) -> int:
    hash_value = 0
    for char in s:
        hash_value = (hash_value * 31 + ord(char)) % (59275)
    return hash_value


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
      boards_per_batch (int): Number of boards per batch.
      num_files_to_skip: Optional number of TPG files to be skipped by this worker (to avoids reprocessing files already processed)
      test (bool): If the Exec test flag is enabled.
  """
  def __init__(self, root_dir, 
               batch_size: int, 
               wdl_smoothing : bool, 
               rank : int, 
               world_size : int, 
               num_workers : int, 
               boards_per_batch : int, 
               num_files_to_skip : int = 0,
               test : bool = False):

    self.root_dir = root_dir
    self.batch_size = batch_size
    self.wdl_smoothing = wdl_smoothing
    self.num_workers = num_workers
    self.generator = self.item_generator()
    self.boards_per_batch = boards_per_batch
    self.test = test
    
    # Get the list of files in the specified directory and select the subset appropriate for this worker.
    all_files = fnmatch.filter(os.listdir(root_dir), '*.zst')
    all_files.sort(key=lambda f: stable_str_hash(f))  # deterministic shuffle
    assert len(all_files) > num_files_to_skip + num_workers, "Trying to skip more files than available"
    all_files = all_files[num_files_to_skip:]

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
    BYTES_PER_POS = 9250 # fixed size of record structure (TPGRecord.TOTAL_BYTES)
    POS_PER_BLOCK = 24576//2 # read this many positions per loop iteration (somewhat arbitrary, each block about 115MB)
    BYTES_PER_BLOCK = POS_PER_BLOCK * BYTES_PER_POS

    if self.worker_id is None:
      print('ERROR: worker_id not initialized')
      exit()

    # Reduce files to be only the files that this worker is responsible for.
    assert self.worker_id >= 0, "Worker ID expected to have been be set before calling item_generator" 
    self.files = [file for index, file in enumerate(self.files) if index % self.num_workers == self.worker_id]      

    wdl_smoothing_transform = np.array([
        [1-self.wdl_smoothing, self.wdl_smoothing*0.75, self.wdl_smoothing*0.25],
        [self.wdl_smoothing*0.5, 1-self.wdl_smoothing, self.wdl_smoothing*0.5],
        [self.wdl_smoothing*0.25, self.wdl_smoothing*0.75, 1-self.wdl_smoothing]])

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

              if (self.wdl_smoothing == 0.5):
                assert 1==2, "wdl_smoothing == 0.5 not supported"

              # Read sequence of fields (see TPGRecord.cs)

              wdl_nondeblundered = np.ascontiguousarray(this_batch[:, offset : offset + 3*4]).view(dtype=np.float32).reshape(-1, 3)
              offset+= 3 * 4
              if (self.wdl_smoothing > 0):
                wdl_nondeblundered = np.matmul(wdl_nondeblundered, wdl_smoothing_transform)

              wdl_deblundered = np.ascontiguousarray(this_batch[:, offset : offset + 3*4]).view(dtype=np.float32).reshape(-1, 3)
              if (self.wdl_smoothing > 0):
                wdl_deblundered = np.matmul(wdl_deblundered, wdl_smoothing_transform)

              offset+= 3 * 4

              wdl_q = np.ascontiguousarray(this_batch[:, offset : offset + 3*4]).view(dtype=np.float32).reshape(-1, 3)
              offset+= 3 * 4
              if (self.wdl_smoothing > 0):
                wdl_q = np.matmul(wdl_q, wdl_smoothing_transform)

              played_q_suboptimality = np.ascontiguousarray(this_batch[:, offset : offset + 1*4]).view(dtype=np.float32).reshape(-1, 1)
              offset+= 1 * 4
           
              #ply_next_square_move = np.ascontiguousarray(this_batch[:, offset : offset + 64 * 1]).view(dtype=np.byte).reshape(-1, 64).astype(DTYPE)
              offset+= 56

              uncertainty_policy = np.ascontiguousarray(this_batch[:, offset : offset + 1*4]).view(dtype=np.float32).reshape(-1, 1)
              uncertainty_policy = np.abs(uncertainty_policy)
              offset+= 1 * 4

              mlh = np.ascontiguousarray(this_batch[:, offset : offset + 1*4]).view(dtype=np.float32).reshape(-1, 1)
              mlh = np.square(mlh / 0.1) # undo preprocessing
              mlh = mlh / 100.
              offset+= 1 * 4

              uncertainty = np.ascontiguousarray(this_batch[:, offset : offset + 1*4]).view(dtype=np.float32).reshape(-1, 1)
              uncertainty = np.abs(uncertainty)
              offset+= 1 * 4

              q_deviation_lower = np.ascontiguousarray(this_batch[:, offset : offset + 1*2]).view(dtype=np.float16).reshape(-1, 1)
              offset+= 1 * 2
              q_deviation_upper = np.ascontiguousarray(this_batch[:, offset : offset + 1*2]).view(dtype=np.float16).reshape(-1, 1)
              offset+= 1 * 2

              policy_index_in_parent = np.ascontiguousarray(this_batch[:, offset : offset + 1*2]).view(dtype=np.int16).reshape(-1, 1)
              offset+= 1 * 2
 
              policies_indices = np.ascontiguousarray(this_batch[:, offset : offset + MAX_MOVES*2]).view(dtype=np.int16).reshape(-1, MAX_MOVES)
              # much faster, but tries to reinitialize CUDA and fails:
              #   policies = torch.from_numpy(np.ascontiguousarray(this_batch[:, offset : offset + 1858*2]).view(dtype=np.float16)).cuda().reshape(-1,1858)
              offset+= MAX_MOVES * 2
              
              policies_values = np.ascontiguousarray(this_batch[:, offset : offset + MAX_MOVES*2]).view(dtype=np.float16).reshape(-1, MAX_MOVES)
              offset+= MAX_MOVES * 2

              SIZE_SQUARE = 137
              squares = np.ascontiguousarray(this_batch[:, offset : offset + 64 * SIZE_SQUARE * 1]).view(dtype=np.byte).reshape(-1, 64, SIZE_SQUARE).astype(DTYPE)
              DIVISOR = 100
              squares = np.divide(squares, DIVISOR).astype(DTYPE)
              offset+= 64 * SIZE_SQUARE

              assert(offset == BYTES_PER_POS)

              yield  ((policies_indices, policies_values, wdl_deblundered, wdl_q, mlh, uncertainty, 
                       wdl_nondeblundered, q_deviation_lower, q_deviation_upper, squares,policy_index_in_parent, played_q_suboptimality,
                       uncertainty_policy))


  def __getitem__(self, idx):
    batch = next(self.generator)
    policies_indices = batch[0]
    policies_values = batch[1]
    wdl_deblundered = batch[2]
    wdl_q = batch[3]
    mlh = batch[4]
    uncertainty = batch[5]
    wdl_nondeblundered = batch[6]
    q_deviation_lower = batch[7]
    q_deviation_upper = batch[8]
    squares = batch[9]
    policy_index_in_parent = batch[10]
    played_q_suboptimality = batch[11]
    uncertainty_policy = batch[12]
    
    policies_indices = torch.tensor(policies_indices, dtype=torch.int64).reshape(self.batch_size, MAX_MOVES)
    policies_values  = torch.tensor(policies_values, dtype=torch.float16).reshape(self.batch_size, MAX_MOVES)

    # TO DO: do this on GPU?
    policies = torch.zeros(self.batch_size, 1858, dtype=torch.float16)
    policies.scatter_(1, policies_indices, policies_values)

   
    def create_filtered_dict(mod_value):
      # Function to filter tensor elements with indices modulo boards_per_batch equal to mod_value
      def filter_tensor(tensor, mod_value):
          indices = torch.arange(len(tensor))
          filtered_indices = indices[indices % self.boards_per_batch == mod_value]
          return tensor[filtered_indices]

      # Creating the new dictionary with filtered tensors
      filtered_dict = {
          'policies': filter_tensor(policies, mod_value),
          'wdl_deblundered': filter_tensor(torch.tensor(wdl_deblundered), mod_value),
          'wdl_q': filter_tensor(torch.tensor(wdl_q), mod_value),
          'mlh': filter_tensor(torch.tensor(mlh), mod_value),
          'unc': filter_tensor(torch.tensor(uncertainty), mod_value),
          'wdl_nondeblundered': filter_tensor(torch.tensor(wdl_nondeblundered), mod_value),
          'q_deviation_lower': filter_tensor(torch.tensor(q_deviation_lower), mod_value).to(torch.float32),
          'q_deviation_upper': filter_tensor(torch.tensor(q_deviation_upper), mod_value).to(torch.float32),
          'squares': filter_tensor(torch.tensor(squares), mod_value),
          'policy_index_in_parent': filter_tensor(torch.tensor(policy_index_in_parent), mod_value),
          'played_q_suboptimality': filter_tensor(torch.tensor(played_q_suboptimality), mod_value),
          'uncertainty_policy': filter_tensor(torch.tensor(uncertainty_policy), mod_value)
      }  
      return filtered_dict
    
    return [create_filtered_dict(i) for i in range(self.boards_per_batch)]


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

