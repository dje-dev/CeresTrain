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

using System.Collections.Generic;
using System.Text.RegularExpressions;

using Renci.SshNet;
using Renci.SshNet.Sftp;
using Renci.SshNet.Common;

using Ceres.Base.Misc;
using Ceres.Base.Benchmarking;

#endregion

namespace CeresTrain.Utils
{
  /// <summary>
  /// Manages SSH client features for executing remote commands 
  /// or downloading files from a specified host via SSH.
  /// </summary>
  public class SSHClient : IDisposable
  {
    /// <summary>
    /// Host to which connection is being made.
    /// </summary>
    public readonly string Host;

    /// <summary>
    /// Name of user to be used for remote login.
    /// </summary>
    public readonly string UserName;

    /// <summary>
    /// Name of private key file containing SSH keys on local machine.
    /// </summary>
    public readonly string PrivateKeyFile;

    /// <summary>
    /// Optional pass phrase to be used in conjunction with key.
    /// </summary>
    public readonly string PassPhrase;

    /// <summary>
    /// Underlying SftpClient object.
    /// </summary>
    public SftpClient SFTClient;

    /// <summary>
    /// Underlying SSHClient.
    /// </summary>
    public SshClient SSHExecClient;


    /// <summary>
    /// Constuctor for SSH download from specified host/username.
    /// </summary>
    /// <param name="host"></param>
    /// <param name="username"></param>
    public SSHClient(string host, string username) 
      : this (host, username, GetPrivateKeyFile(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)))
    {
    }


    /// <summary>
    /// Constructor for SSH download from specified host.
    /// </summary>
    /// <param name="host"></param>
    /// <param name="username"></param>
    /// <param name="privateKeyFile"></param>
    /// <param name="passPhrase"></param>
    public SSHClient(string host, string username, string privateKeyFile, string passPhrase = "")
    {
      Host = host;
      UserName = username;
      PrivateKeyFile = privateKeyFile;

      if (passPhrase == "")
      {
        string envPassPharse = Environment.GetEnvironmentVariable("SSH_PASSPHRASE");
        PassPhrase = envPassPharse == null ? "" : envPassPharse;
      }
      else
      {
        PassPhrase = passPhrase;
      }
    }


    /// <summary>
    /// Returns name of private key file in ~/.ssh directory.
    /// </summary>
    /// <param name="homePath"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    static string GetPrivateKeyFile(string homePath)
    {
      string[] files = Directory.GetFiles(Path.Combine(homePath, ".ssh"), "id_rsa*");
      if (files.Length == 0)
      {
        throw new Exception("No private key files found in ~/.ssh");
      }
      return files[0];
    }


    /// <summary>
    /// Returns List of name of files existing in a specified remote directory 
    /// matching an optional regular expression filter (such as ".*\.zst$").
    /// </summary>
    /// <param name="remoteDir"></param>
    /// <param name="filenameFilter"></param>
    /// <param name="useFullNames"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public List<string> FilesInDirectory(string remoteDir, string filenameFilter = null, bool useFullNames = true)
    {
      CheckSFTConnected();

      try
      {
        IEnumerable<ISftpFile> files = SFTClient.ListDirectory(remoteDir);
        List<string> fileNames = new();

        foreach (SftpFile file in files)
        {
          if (!file.IsDirectory && !file.IsSymbolicLink && (filenameFilter == null || Regex.IsMatch(file.Name, filenameFilter)))
          {
            fileNames.Add(useFullNames ? file.FullName : file.Name);
          }
        }

        return fileNames;
      }
      catch (Exception ex)
      {
        throw new Exception($"Exception in SSHClient.FilesInDirectory for {remoteDir}: {ex.Message}");
      }
    }


    /// <summary>
    /// Downloads set of files with specified names from remote host into a local directory
    /// (base file name on local system will be same as on host system).
    /// </summary>
    /// <param name="remoteDir"></param>
    /// <param name="localPath"></param>
    /// <param name="filenameFilter"></param>
    /// <param name="fileDownloadedCallback"></param>
    public List<string> DownloadFilesInDirectory(string remoteDir, string localPath,
                                                 string filenameFilter = null,
                                                 Action<string> fileDownloadedCallback = null,
                                                 bool skipFileIfExistsLocally = false)
    {
      List<string> files = FilesInDirectory(remoteDir, filenameFilter, true);
      foreach (string remoteFileFullName in files)
      {
        FileInfo remoteFileInfo = new FileInfo(remoteFileFullName);
        string localFullFileName = Path.Combine(localPath, remoteFileInfo.Name);
        DownloadFile(remoteFileFullName, localFullFileName, skipFileIfExistsLocally);
        fileDownloadedCallback?.Invoke(remoteFileFullName);
      }
      return files;
    }


    /// <summary>
    /// Downloads set of files with specified names from remote host into a local directory
    /// (base file name on local system will be same as on host system).
    /// </summary>
    /// <param name="remoteDir"></param>
    /// <param name="localDir"></param>
    /// <param name="fileNames"></param>
    /// <param name="skipIfExistsLocally"></param>
    /// <param name="fileDownloadedCallback"></param>
    public string[] DownloadFiles(string remoteDir, string localDir,
                                  string[] fileNames,
                                  bool skipIfExistsLocally = false,
                                  Action<string> fileDownloadedCallback = null)
    {
      // Create local directory if not already present.
      if (!Directory.Exists(localDir))
      {
        Directory.CreateDirectory(localDir);
      }

      if (fileNames == null)
      {
        fileNames = FilesInDirectory(remoteDir, useFullNames: false).ToArray();
      }

      foreach (string fileName in fileNames)
      {
        string localFullFN = Path.Combine(localDir, fileName);
        string remoteFullFN = remoteDir + "/" + fileName; // Use forward slash so works on both Windows and Linux

        DownloadFile(remoteFullFN, localFullFN, skipIfExistsLocally);
        fileDownloadedCallback?.Invoke(localFullFN);
      }

      return fileNames;
    }


    /// <summary>
    /// Returns List of name of files existing in a specified remote directory 
    /// matching an optional regular expression filter (such as ".*\.zst$").
    /// </summary>
    /// <param name="remotePath"></param>
    /// <param name="localPath"></param>
    /// <param name="skipIfExistsLocally"></param>
    /// <param name="logToConsole"></param>
    /// <exception cref="Exception"></exception>
    public void DownloadFile(string remotePath, string localPath, bool skipIfExistsLocally = false, bool logToConsole = true)
    {
      if (skipIfExistsLocally && File.Exists(localPath))
      {
        // Already found locally.
        return;
      }

      CheckSFTConnected();

      TimingStats timingStats = new TimingStats();
      using (new TimingBlock(timingStats, TimingBlock.LoggingType.None))
      {
        using (FileStream fileStream = new(localPath, FileMode.Create, FileAccess.Write))
        {
          try
          {
            SFTClient.DownloadFile(remotePath, fileStream);
          }
          catch (SftpPathNotFoundException sfne)
          {
            throw new Exception($"Remote file not found: {remotePath}");
          }
        }
      }

      if (logToConsole)
      {
        Console.WriteLine($"Downloaded {localPath} size {new FileInfo(localPath).Length / (1024.0 * 1024)} mb from {Host} in {timingStats.ElapsedTimeSecs:n2} seconds.");
      }
    }


    /// <summary>
    /// Uploads specified local file to the remote host with specified remote file path.
    /// </summary>
    /// <param name="localPath"></param>
    /// <param name="remotePath"></param>
    /// <param name="logToConsole"></param>
    public void UploadFile(string localPath, string remotePath, bool skipIfExistsRemotely = false, bool logToConsole = true)
    {
      if (skipIfExistsRemotely && SFTClient.Exists(remotePath))
      {
        // Already found on remote.
        return;
      }

      CheckSFTConnected();

      TimingStats timingStats = new TimingStats();
      using (new TimingBlock(timingStats, TimingBlock.LoggingType.None))
      {
        using (FileStream fileStream = new(localPath, FileMode.Open, FileAccess.Read))
        {
          SFTClient.UploadFile(fileStream, remotePath);
        }
      }

      if (logToConsole)
      {
        Console.WriteLine($"Uploaded {localPath} size {new FileInfo(localPath).Length / (1024.0 * 1024)} mb to {Host} in {timingStats.ElapsedTimeSecs:n2} seconds.");
      }
    }


    /// <summary>
    /// Establishes SFTP connection if not already established.
    /// </summary>
    private void CheckSFTConnected()
    {
      if (SFTClient == null)
      {
        SFTClient = GetSftpClient();
      }
    }


    /// <summary>
    /// Returns if the private key requires a passphrase.
    /// </summary>
    /// <param name="privateKeyPath"></param>
    /// <returns></returns>
    public static bool IsPassphraseRequired(string privateKeyPath)
    {
      try
      {
        // Try to load the private key without a passphrase
        PrivateKeyFile keyFile = new PrivateKeyFile(privateKeyPath);
        return false;
      }
      catch (SshPassPhraseNullOrEmptyException)
      {
        return true;
      }
    }


    /// <summary>
    /// Checks if SSH client is connected and if not, creates a new one.
    /// </summary>
    public void CheckSSHClientConnected()
    {
      if (SSHExecClient == null)
      {
        SSHExecClient = new SshClient(GetConnectInfo());
      }
    }


    /// <summary>
    /// Executes specified command on remote host and returns the output.
    /// </summary>
    /// <param name="command"></param>
    /// <returns></returns>
    public string ExecuteSSHCommand(string command)
    {
      CheckSSHClientConnected();

      SshCommand cmd_ls = SSHExecClient.CreateCommand(command);
      return cmd_ls.Execute();
    }


    public void ExecuteSSHCommandStream(IEnumerable<(string command,
                                        Predicate<string> shouldStopPredicate)> commands,
                                        bool shouldEcho)
    {
      CheckSSHClientConnected();

      string remoteShellName = $"RemoteCommand_{Environment.MachineName}_{Environment.UserName}";
      using (ShellStream shellStream = SSHExecClient.CreateShellStream(remoteShellName, 80, 24, 800, 600, 1024))
      {
        foreach (var command in commands)
        {
          shellStream.WriteLine(command.command);
          while (true)
          {
            string line = shellStream.ReadLine();

            // Possibly break out of loop if command indicates we should stop.
            if (line == null || command.shouldStopPredicate(line))
            {
              break;
            }

            if (shouldEcho)
            {
              Console.ForegroundColor = ConsoleColor.Red;
              Console.WriteLine(command.command);
              Console.ForegroundColor = ConsoleColor.Yellow;
              Console.WriteLine(line);
              Console.ResetColor();
            }
          }
        }
      }
    }


    private ConnectionInfo GetConnectInfo()
    {
      PrivateKeyFile[] keyFiles = new[] { new PrivateKeyFile(PrivateKeyFile, PassPhrase) };

      AuthenticationMethod[] methods = new AuthenticationMethod[]
      {
        new PrivateKeyAuthenticationMethod(UserName, keyFiles),
        PassPhrase is not null ? new PasswordAuthenticationMethod(UserName, PassPhrase) : null
      };

      return new ConnectionInfo(Host, UserName, methods);
    }


    /// <summary>
    /// Constructs and return SftpClient to be used for connections.
    /// </summary>
    /// <returns></returns>
    private SftpClient GetSftpClient()
    {
      SftpClient client;

      // For unknown reasons, valid client connect attempts occasionally fail,
      // therefore try multiple times if necessary.
      const int MAX_TRIES = 5;
      for (int i = 0; i < MAX_TRIES; i++)
      {
        try
        {
          client = TryGetSftpClient();
          return client;
        }
        catch (Exception e)
        {
        }
      }
      throw new Exception("SftpClient unable to connect to host " + Host);
    }


    private SftpClient TryGetSftpClient()
    {
      SftpClient client = new SftpClient(GetConnectInfo());
      client.Connect();
      return client;
    }


    /// <summary>
    /// Queries user for passphrase from the console and returns.
    /// </summary>
    /// <returns></returns>
    public static string PassphrasesPromptedFromUser() => ConsoleUtils.ConsoleReadStringHidden("Enter SSH passphrase");


    /// <summary>
    /// Disposes of associated objects.
    /// </summary>
    public void Dispose()
    {
      SFTClient?.Disconnect();
      SFTClient?.Dispose();
      SFTClient = null;
    }

  }

}
