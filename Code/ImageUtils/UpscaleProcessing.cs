using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using Cupscale.IO;
using ImageMagick;
using Paths = Cupscale.IO.Paths;

namespace Cupscale
{
	internal class UpscaleProcessing
	{
		public enum Format
		{
			PngOpti,
			PngFast,
			JpegHigh,
			JpegMed,
			WeppyHigh,
			WeppyLow,
			BMP,
			TGA,
			DDS
		}

		//public static Button upscaleBtn;

		public static void ChangeOutputExtensions(string newExtension)
		{
			string path = Paths.imgOutPath;
			DirectoryInfo d = new DirectoryInfo(path);
			FileInfo[] files = d.GetFiles("*", SearchOption.AllDirectories);
			FileInfo[] array = files;
			foreach (FileInfo file in array)
			{
				file.MoveTo(file.FullName.Substring(0, file.FullName.Length - 4));
			}
			FileInfo[] array2 = files;
			foreach (FileInfo file2 in array2)
			{
				file2.MoveTo(Path.ChangeExtension(file2.FullName, newExtension));
			}
		}

		public static async Task ConvertImagesToOriginalFormat()
		{
			string path = Paths.imgOutPath;
			DirectoryInfo d = new DirectoryInfo(path);
			FileInfo[] files = d.GetFiles("*", SearchOption.AllDirectories);
			FileInfo[] array = files;
			foreach (FileInfo file in array)
			{
				file.MoveTo(file.FullName.Substring(0, file.FullName.Length - 4));
			}
			FileInfo[] array2 = files;
			foreach (FileInfo file2 in array2)
			{
				if (GetTrimmedExtension(file2) == "png")
				{
					break;
				}
				Format format = Format.PngOpti;
				if (GetTrimmedExtension(file2) == "jpg" || GetTrimmedExtension(file2) == "jpeg")
				{
					format = Format.JpegHigh;
				}
				if (GetTrimmedExtension(file2) == "webp")
				{
					format = Format.WeppyHigh;
				}
				if (GetTrimmedExtension(file2) == "bmp")
				{
					format = Format.BMP;
				}
				if (GetTrimmedExtension(file2) == "tga")
				{
					format = Format.TGA;
				}
				if (GetTrimmedExtension(file2) == "dds")
				{
					format = Format.DDS;
				}
				await ConvertImage(file2.FullName, format, false, false, false);
				await Task.Delay(1);
			}
		}

		private static string GetTrimmedExtension(FileInfo file)
		{
			return file.Extension.ToLower().Replace(".", "");
		}

		public static async Task ConvertImages(string path, Format format, bool removeAlpha = false, bool preprocess = false, bool appendExtension = false, bool delSource = true)
		{
			DirectoryInfo d = new DirectoryInfo(path);
			FileInfo[] files = d.GetFiles("*", SearchOption.AllDirectories);
			FileInfo[] array = files;
			foreach (FileInfo file in array)
			{
				Logger.Log("Converting " + file.Name + " to " + format.ToString() + ", appendExtension = " + appendExtension);
				await ConvertImage(file.FullName, format, removeAlpha, appendExtension, delSource);
				await Task.Delay(1);
			}
		}

		public static async Task ConvertImage(string path, Format format, bool fillAlpha, bool appendExtension, bool deleteSource = true)
		{
			MagickImage img = new MagickImage(path);
			Logger.Log("Converting: " + img.ToString() + " - Target Format: " + format.ToString() + " - DeleteSource: " + deleteSource);
			string text = "png";
			if (format == Format.PngOpti)
			{
				img.Format = (MagickFormat)171;
				img.Quality = (80);
			}
			if (format == Format.PngFast)
			{
				img.Format = (MagickFormat)171;
				img.Quality = (20);
			}
			if (format == Format.JpegHigh)
			{
				img.Format = (MagickFormat)105;
				img.Quality = (95);
				text = "jpg";
			}
			if (format == Format.JpegMed)
			{
				img.Format = (MagickFormat)105;
				img.Quality = (80);
				text = "jpg";
			}
			if (format == Format.WeppyHigh)
			{
				img.Format = (MagickFormat)239;
				img.Quality = (92);
				text = "webp";
			}
			if (format == Format.WeppyLow)
			{
				img.Format = (MagickFormat)239;
				img.Quality = (80);
				text = "webp";
			}
			if (format == Format.BMP)
			{
				img.Format = (MagickFormat)17;
				text = "bmp";
			}
			if (format == Format.TGA)
			{
				img.Format = (MagickFormat)218;
				text = "tga";
			}
			if (format == Format.DDS)
			{
				img.Format = (MagickFormat)43;
				text = "dds";
			}
            if (fillAlpha)
            {
				img.ColorAlpha(new MagickColor("#" + Config.Get("alphaBgColor")));
			}
			if (appendExtension)
			{
				string extension = Path.GetExtension(path);
				Logger.Log("Appending old extension; writing image to " + Path.ChangeExtension(path, null) + extension + "." + text);
				img.Write(Path.ChangeExtension(path, null) + extension + "." + text);
				if (deleteSource && !(Path.ChangeExtension(path, null) + extension + "." + text == path))
				{
					Logger.Log("Deleting source file: " + path);
					File.Delete(path);
				}
			}
			else
			{
				img.Write(Path.ChangeExtension(path, text));
				Logger.Log("Writing image to " + Path.ChangeExtension(path, text));
				if (deleteSource && !(Path.ChangeExtension(path, text) == path))
				{
					Logger.Log("Deleting source file: " + path);
					File.Delete(path);
				}
			}
			await Task.Delay(1);
		}
	}
}
