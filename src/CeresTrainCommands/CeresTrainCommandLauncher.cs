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
using System.CommandLine;

using Ceres.Base.Misc;
using Ceres.Chess;

using CeresTrain.Examples;
using CeresTrain.UserSettings;
using CeresTrain.CeresTrainDefaults;
using CeresTrain.Tasks;

#endregion

namespace CeresTrain.TrainCommands
{
    /// <summary>
    /// Runs console session (with command line arguments) for CeresTrain,
    /// allowing for training, evaluation, and running of Ceres networks.
    /// </summary>
    public static class CeresTrainCommandLauncher
  {
    static RootCommand rootCommand;

    static Option<string> configOption;
    static Option<long> numPosOption;
    static Option<int> numTPGSetsOption;
    static Option<string> piecesOption;
    static Option<string> netSpecificationOption;
    static Option<string> netSpecificationOptionalOption;
    static Option<string> netSpecificationFillinOption;
    static Option<string> tpgDirOption;
    static Option<string> tarDirOption;
    static Option<string> packedZSTDirOption;
    static Option<bool> verboseOption;
    static Option<string> hostOption;
    static Option<string> searchLimitOptionDefaultBV;
    static Option<string> searchLimitOption;
    static Option<int[]> devicesOption;
    static Option<string> epdOrPgnFnOption;
    static Option<string> epdOrPgnOutputFileNameOption;

    static Command infoCommand;
    static Command trainCommand;
    static Command evalCommand;
    static Command tournCommand;
    static Command evalLC0Command;
    static Command generateEndgameTPGCommand;
    static Command extractPositionsCommand;
    static Command uciCommand;
    static Command initCommand;

    static Command generateTPGCommand;
    static Command convertTARToPackedZSTCommand;

    /// <summary>
    /// Starts a console session for CeresTrain, reading command line arguments and executing the appropriate command.
    /// </summary>
    /// <param name="args"></param>
    /// <exception cref="Exception"></exception>
    public static void LaunchProcessCommandLine(string[] args)
    {
      rootCommand = new RootCommand("CeresTrain - command line executor for CeresTrain to train/evaluate/run Ceres networks.");

      configOption = new Option<string>("--config", "Configuration name") { IsRequired = true };
      numPosOption = new Option<long>("--num-pos", () => 2048, "Number of positions") { };
      numTPGSetsOption = new Option<int>("--num-tpg-sets", () => 1, "Number of sets of TPG positions to generate (~200mm positions per set)") { };
      piecesOption = new Option<string>("--pieces", "Chess pieces (e.g. KRPkrp)") { IsRequired = true };
      netSpecificationOption = new Option<string>("--net-spec", "LC0 network specification in Ceres format, e.g. LC0:703810") { IsRequired = true };
      netSpecificationOptionalOption = new Option<string>("--net-spec", "LC0 network specification used to compare performance against (or null for tablebase)") { IsRequired = false };
      netSpecificationFillinOption = new Option<string>("--net-spec-fillin", "LC0 network specification of network to use for noncovered positions") { IsRequired = false };
      tpgDirOption = new Option<string>("--tpg-dir", "Directory containing TPG training data files") { IsRequired = false };
      tarDirOption = new Option<string>("--tar-dir", "Directory containing TAR training data files") { IsRequired = false };
      packedZSTDirOption = new Option<string>("--zst-dir", "Directory containing packed ZST training data files") { IsRequired = true };
      verboseOption = new Option<bool>("--verbose", "If verbose information should be sent to Console (true of false).");
      hostOption = new Option<string>("--host", "Name of host (or WSL) on which to execute command.") { IsRequired = false };
      devicesOption = new Option<int[]>("--devices", "List of indices of devices to use.") { IsRequired = false };
      searchLimitOptionDefaultBV = new Option<string>("--search-limit", () => "bv", "Search limit to use (e.g. BV for best value).") { IsRequired = false };
      searchLimitOption = new Option<string>("--search-limit", () => null, "Search limit to use if tournament is to be r.") { IsRequired = false };
      epdOrPgnFnOption = new Option<string>("--pos-fn", "EPD or PGN file name used to source positions") { IsRequired = false };
      epdOrPgnOutputFileNameOption = new Option<string>("--pos-out-fn", "Name of PGN or EPD file from which to extract positions") { IsRequired = true };    

      // Add commands
      initCommand = new Command("init", "Initializes new config with default values.                     [config]") { configOption };
      infoCommand = new Command("info", "Display information about a configuration.                      [config]") { configOption };
      trainCommand = new Command("train", "Start training using a configuration.                           [config] [pieces] [num-pos] [tpg-dir] [host] [devices]") { configOption, piecesOption, numPosOption, tpgDirOption, hostOption, devicesOption };
      evalCommand = new Command("eval", "Evaluate accuracy of last trained net.                          [config] [pieces] [num-pos] [pos-fn] [verbose] [net-spec]") { configOption, piecesOption, numPosOption, verboseOption, netSpecificationOptionalOption, epdOrPgnFnOption };
      tournCommand = new Command("tourn", "Run tournament between net and specified LC0 net (or TB).       [config] [pieces] [num-pos] [pos-fn] [verbose] [net-spec] [search-limit]") { configOption, piecesOption, numPosOption, epdOrPgnFnOption, verboseOption, netSpecificationOptionalOption, searchLimitOptionDefaultBV };
      uciCommand = new Command("uci", "Launch trained net with specified configuration as UCI engine.  [config] [pieces] [net-spec-fillin]") { configOption, piecesOption, netSpecificationFillinOption };
      evalLC0Command = new Command("eval-lc0", "Evaluate vs LC0 with specific pieces and network.               [pieces] [net-spec] [num-pos] [search-limit] [pos-fn] [verbose]") { piecesOption, netSpecificationOption, numPosOption, searchLimitOption, epdOrPgnFnOption, verboseOption };
      extractPositionsCommand = new Command("extract-pos", "Generate EPD/PGN file with positions from specified PGN/EPD     [pieces] [num-pos] [pos-fn] [pos-out-fn]") { piecesOption, numPosOption, epdOrPgnFnOption, epdOrPgnOutputFileNameOption };
      generateEndgameTPGCommand = new Command("gen-endgame-tpg", "Generate TPG files with positions from specified pieces or \"*\"  [pieces] [num-pos] [tar-dir] [tpg-dir]") { piecesOption, numPosOption, tarDirOption, tpgDirOption };
      generateTPGCommand = new Command("gen-tpg", "Generate TPG files from TAR files.                              [tar-dir] [tpg-dir] [num-sets]") { tarDirOption, tpgDirOption, numTPGSetsOption };
      convertTARToPackedZSTCommand = new Command("convert-tar-to-zst", "Convert TAR files to packed ZST files.                          [tar-dir] [zst-dir]") { tarDirOption, packedZSTDirOption };

      rootCommand.AddCommand(initCommand);
      rootCommand.AddCommand(infoCommand);
      rootCommand.AddCommand(trainCommand);
      rootCommand.AddCommand(evalCommand);
      rootCommand.AddCommand(tournCommand);
      rootCommand.AddCommand(uciCommand);
      rootCommand.AddCommand(evalLC0Command);
      rootCommand.AddCommand(extractPositionsCommand);
      rootCommand.AddCommand(generateEndgameTPGCommand);
      rootCommand.AddCommand(generateTPGCommand);
      rootCommand.AddCommand(convertTARToPackedZSTCommand);

      InstallCommandHandlers();

      rootCommand.Invoke(args);
    }


    /// <summary>
    /// Installs the handlers associated with all of the available commands.
    /// </summary>
    /// <exception cref="Exception"></exception>
    static void InstallCommandHandlers()
    {
      string configsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "configs");
      string resultsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "results");

      Console.WriteLine();

      initCommand.SetHandler((configID) =>
      {
        if (configID == null)
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Red, CeresTrainCommandUtils.NO_CONFIG_ERR_STR);
          throw new Exception();
        }

        CeresTrainCommands.ProcessInitCommand(in CeresTrainDefault.DEFAULT_CONFIG_TRAINING, configID);
      }, configOption);

      extractPositionsCommand.SetHandler((string pieces, long numPos, string epdOrPgnFnOption, string epdOrPgnOutputFileNameOption) =>
      {
        CeresTrainCommands.ExtractToEPD(epdOrPgnFnOption, new PieceList(pieces), epdOrPgnOutputFileNameOption, (int)numPos);
      }, piecesOption, numPosOption, epdOrPgnFnOption, epdOrPgnOutputFileNameOption);


      infoCommand.SetHandler((configID) =>
      {
        CeresTrainCommands.ProcessInfoCommand(configID, configsDir);
      }, configOption);


      uciCommand.SetHandler((configID, piecesStr, netSpecFillInStr) =>
      {
        CeresTrainCommands.ProcessUCICommand(configID, piecesStr, configsDir, netSpecFillInStr);
      }, configOption, piecesOption, netSpecificationFillinOption);


      convertTARToPackedZSTCommand.SetHandler((sourceDir, targetDir) =>
      {
        TPGConvertFromTAR.GeneratePackedZSTFromTARs(sourceDir, targetDir);
      }, tarDirOption, packedZSTDirOption); 


      generateTPGCommand.SetHandler((sourceDir, targetDir, numSets) =>
      {
        TPGConvertFromTAR.GenerateTPG(sourceDir, targetDir, numSets, "Converted using TPGConvertFromTAR.GenerateTPG");
      }, tarDirOption, tpgDirOption, numTPGSetsOption);  


      generateEndgameTPGCommand.SetHandler((piecesStr, numPos, tarDirectory, outDirectory) =>
      {
        if (tarDirectory == null)
        {
          CeresNetEvaluation.GenerateTPGFilesFromRandomTablebasePositions(piecesStr, numPos, outDirectory);
        }
        else
        {
          CeresNetEvaluation.GenerateTPGFilesFromLC0TrainingData(piecesStr, numPos, tarDirectory, outDirectory);
        }
      }, piecesOption, numPosOption, tarDirOption, tpgDirOption);


      trainCommand.SetHandler((configID, piecesStr, numPos, tpgDir, hostName, devices) =>
      {
        CeresTrainCommands.ProcessTrainCommand(configID, piecesStr, numPos, hostName, tpgDir, devices, null);
      }, configOption, piecesOption, numPosOption, tpgDirOption, hostOption, devicesOption);


      evalCommand.SetHandler((configID, piecesStr, numPos, epdOrPgnFN, verbose, netSpecification) =>
      {
        CeresTrainCommands.RunEvalOrTournament(configID, piecesStr, numPos, epdOrPgnFN, verbose, netSpecification, false, configsDir, false, default);
      }, configOption, piecesOption, numPosOption, epdOrPgnFnOption,  verboseOption, netSpecificationOptionalOption);


      tournCommand.SetHandler((configID, piecesStr, numPos, epdOrPgnFN, verbose, compareLC0NetSpec, searchLimitSpec) =>
      {
        SearchLimit searchLimit = SearchLimitSpecificationString.Parse(searchLimitSpec);
        bool enableOpponentTB = compareLC0NetSpec == null;
        CeresTrainCommands.RunEvalOrTournament(configID, piecesStr, numPos, epdOrPgnFN, verbose, compareLC0NetSpec, enableOpponentTB, configsDir, true, searchLimit);
      }, configOption, piecesOption, numPosOption, epdOrPgnFnOption, verboseOption, netSpecificationOptionalOption, searchLimitOptionDefaultBV);


      evalLC0Command.SetHandler((piecesStr, netID, numPos, searchLimitSpec, epdOrPgnFN, verbose) =>
      {
        SearchLimit searchLimit = searchLimitSpec == null ? default : SearchLimitSpecificationString.Parse(searchLimitSpec);
        CeresTrainCommands.ProcessEvalLC0Command(piecesStr, netID, numPos, epdOrPgnFN, searchLimit, verbose);
      }, piecesOption, netSpecificationOption, numPosOption, searchLimitOption, epdOrPgnFnOption, verboseOption);
    }

  }
}
