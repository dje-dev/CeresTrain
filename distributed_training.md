## Distributed Training with CeresTrain

CeresTrain supports launching a training command (either from the command line or via the API) on a local machine (running either Microsoft Windows or Linux) but specifying that the PyTorch training is to run on another host (or on the WSL on Windows). This can be accomplished by simply specifying additional command line arguments to the train command.

CeresTrain will coordinate with the host over SSH. All status lines emitted by the host will be logged to a file, and a live summary status line on training progress will be show to the user on the Console. 

Note that it is required that SSH connectivity have already been configured with the server including SSL keys to enable ssh remote login without requiring an explicit password.

First a host configuration file must be created on the local computer. It should have the name "CeresTrainHosts.json" and be located in the CeresTrain data directory. There are four required entries. In addition to hostname and username, there must be entries for:
* *CeresTrainPyDir** which indicates the host directory containing the CeresTrain python files. These files can be downloaded to the remote host by cloning the CeresTrain project to that machine, whereafter they will be found in the "CeresTrain\src\CeresTrainPy" directory. 

* *PathToOutputFromHost* which indicates a path to which the artifacts of the training process should be sent (checkpoint files and ONNX conversions thereof). Typically a path the local CeresTrain output directory is used. Here is an example CeresTrainHosts.json file specifying two hosts (one WSL and one a true remote Linux server):

* *DockerLaunchCommand* (optional) which indicates a command to be used to launch the docker container on the remote host. This is only required if the remote host is a Windows machine running Docker. The command should include the docker launch command if Exec.RunInDocker flag is set true in configuration.
```
```
[
  {
    "HostName": "wsl",
    "UserName": "user2",
    "CeresTrainPyDir": "~/dev/CeresTrainPy",
    "PathToOutputFromHost": "/mnt/e/cout",
  },
    {
    "HostName": "server1",
    "UserName": "user1",
    "CeresTrainPyDir": "~/dev/CeresTrainPy",
    "PathToOutputFromHost": "/mnt/deve/cout",
    "DockerLaunchCommand": "docker exec -it upbeat_fermi",
  }
]
``` 

Note that the Python modules which execute on the remote host have a number of Python package dependencies. It will likely be necessary to install some or all of these using pip. For example:
```
pip install zstandard
pip install torch torchvision torchaudio
pip install lightning
pip install einops
pip install tensorboard
pip install onnx
pip install onnxruntime-gpu
pip install onnxconverter
pip install onnxconverter-common
pip install onnxscript
pip install onnxmltools
pip install torchsummary
pip install lightning
```

A data preprocessing step is also required. Unlike local inprocess C# training, it is not feasible  to generate training data "on the fly" because the slow speed of the target language (Python). Therefore it is necessary to use the "gen_tpg" command to pregenerate training data files (in a CeresTrain format called TPG) for use by the training session. For example:

```
CeresTrain gen_tpg --pieces=KPRkrp --num_pos=10000000 --tpg_dir=d:\data\KRPkrp_100mm
```

These TPG files are compressed using the zstandard library developed by Facebook/Meta. Therefore it is necessary the the zstandard library file be installed on both local and remote computers. On Linux, pip install is possible. On Windows it is necessary to visit the [zstd github page](https://github.com/facebook/zstd), download the appropriate zip file, extract the "libzstd.dll" and make sure it is located on the system path.

The "host" argument can now be used on the training command to indicate the hostname (computer name) of the target computer (or alternatively the special name WSL which maps to the default local WSL installation on a Windows system):
```
--host=server1
```

The "tpg-dir" argument must also be provided to specify the location of the directory containing the TPG files (from the perspective of the host system). For example, the following assumes that the host computer has mounted the file system on the which the TPG files were generated ("coordinator" in this example) with the name "drived":
```
--tpg-dir=/mnt/coordinator/drived/data
```

Finally, it may be desired to disignate which devices (GPUs) will be used to perform training, for example to use the GPU with index 3:
```
--devices=3
```

