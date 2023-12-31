
## Installing and Configuring CeresTrain

The CeresTrain system is a moderately complex and resource-intensive, having numerous software dependencies and somewhat intense hardware requirements. It must be bulit from source (using two different projects from Github). It has not been tested on systems with less than 32GB of memory and an NVIDIA 2070 graphic card and will work most effectively on more powerful consumer or prosumer configurations.

Note that configuring remote Python based training is yet more complex and requires [additional configuration steps](distributed_training.md).

Eventually a dockerfile will be provided to automate the process of building to run in a container.

The steps below describe how to install and configure on a Windows or Linux system.


1. Make sure the minimum hardware and software requirements are satisfied, including:
- Windows 10+ or Linux operating system
- NVIDIA GPU with at least 8GB of VRAM
- minimum 32GB of system memory
- sufficient disk space for Syzygy tablebases and CeresTrain binaries


2. Install the [Microsoft .NET SDK](https://dotnet.microsoft.com/en-us/download/dotnet/8.0) (version 8.0 or above)


3. Download Syzygy endgame tablebase files (ideally onto an SSD for fast lookups). The sizes of 5-man, 6-man and 7-man tablebases are approximately 1gb, 150gb, and 1800gb, respectively. These files can be downloaded via torrents or from one of the various web sites which host these files. Optionally test files can then be checked for integrity if the md5sum utility program is available. Downloading can be accomplished manually or using bulk downloader utilities. For example: 

```
wget --mirror --no-parent --no-directories -e robots=off http://tablebase.sesse.net/syzygy/3-4-5/
md5sum --check checksum.md5
``````


4. Use git clone to create local repositories of two Ceres projects into the same parent directory. The CeresTrain project files refer to and depend on the Ceres files, and expect them to be located in this relative position. Currently this project dependency is not encoded in the CeresTrain project. Eventually an alternate approach utilizing git submodules or subtrees may instead be adopted.

```
git clone https://github.com/dje-dev/Ceres
git clone https://github.com/dje-dev/CeresTrain
```

5. It should first be confirmed that the Ceres engine works standalone, with reference to the [Ceres installation instructions](https://github.com/dje-dev/Ceres/blob/main/Setup.md). Two entries in this file are essential for work with CeresTrain. The first sets the path to the Syzygy endgame tablebase files. The second sets the path to a directory LC0 neural network files are located. These are (optionally) used for comparison purposes. Examples :
```
  "SyzygyPath": "e:\\sygyzy\\5and6man",
  "DirLC0Networks": "d:\\nets",
```

6. A local directory should be created to contain the files that will be created by CeresTrain. These include configuration and results summary files (in JSON format) as well as artifacts produced by the training process such as training checkpoint files and ONNX network files. 

7. Finally it is necessary to create a file "CeresTrain.json" containing configuration information for CeresTrain. This file can be located in either the user's home directory ("%home% on Windows, or "~" on Linux) or in the working directory from which the CeresTrain executable is launched.

Two entries are required. The first points to the configuration file used by Ceres mentioned in step 5. The second references the output directory described in step 6. For example:

```
{
  "CeresJSONFileName": "c:\\dev\\ceres\\artifacts\\release\\net8.0\\Ceres.json",
  "OutputsDir": "e:\\cout",
}
```

