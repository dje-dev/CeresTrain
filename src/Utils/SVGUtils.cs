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
using System.Text;
using System.Diagnostics;

#endregion

namespace CeresTrain.Utils
{
  /// <summary>
  /// Set of static helper methods relating to generating and viewing SVG files,
  /// in particular for displaying heatmaps over chessboards squares.
  /// </summary>
  public static class SVGUtils
  {
    /// <summary>
    /// Write specified SVG to a file and launches application to view (Windows only).
    /// </summary>
    /// <param name="svgString"></param>
    public static void DisplaySVG(string svgString)
    {
      string svgFileName = Path.GetTempFileName() + ".svg";

      File.WriteAllText(svgFileName, svgString);

      string url = svgFileName.Replace("&", "^&");
      Process.Start(new ProcessStartInfo("cmd", $"/c start {url}") { CreateNoWindow = true });
    }


    /// <summary>
    /// Returns the SVG with an 8x8 heatmap for a given 2D matrix of of values.
    /// </summary>
    /// <param name="matrix8by8"></param>
    /// <returns></returns>
    public static string GenLargeSVGStringForHeatmap8by8FromMatrix(float[,] matrix8by8)
    {
      StringBuilder sb = new();
      sb.AppendLine("<svg width=\"800\" height=\"800\" xmlns=\"http://www.w3.org/2000/svg\">");
      sb.AppendLine("<g>");
      sb.AppendLine("<rect width=\"800\" height=\"800\" style=\"fill:rgb(255,255,255);stroke-width:3;stroke:rgb(0,0,0)\" />");

      for (int i = 0; i < 8; i++)
      {
        for (int j = 0; j < 8; j++)
        {
          float value = matrix8by8[i, j];
          string color = ColorFromValue(value);
          sb.AppendLine($"<rect x=\"{i * 100}\" y=\"{j * 100}\" width=\"100\" height=\"100\" style=\"fill:{color};stroke-width:3;stroke:rgb(0,0,0)\" />");
        }
      }
      sb.AppendLine("</g>");
      sb.AppendLine("</svg>");
      return sb.ToString();
    }


    /// <summary>
    /// Tests SVG generator/viewer for an 8x8 heatmap.
    /// </summary>
    public static void TestDisplaySVGWithRandomData()
    {
      float[,] matrix8by8 = new float[8, 8];
      Random r = new();
      for (int i = 0; i < 8; i++)
      {
        for (int j = 0; j < 8; j++)
        {
          matrix8by8[i, j] = (float)r.NextDouble();
        }
      }
      string svgString = GenLargeSVGStringForHeatmap8by8FromMatrix(matrix8by8);
      DisplaySVG(svgString);
    }


    /// <summary>
    /// Maps a value into a heatmap color.
    /// </summary>
    /// <param name="valueBetween0And1"></param>
    /// <returns></returns>
    static string ColorFromValue(float valueBetween0And1)
    {
      int r = (int)(255 * valueBetween0And1);
      int g = (int)(255 * (1 - valueBetween0And1));
      int b = 0;
      return $"rgb({r},{g},{b})";
    }
  }

}
