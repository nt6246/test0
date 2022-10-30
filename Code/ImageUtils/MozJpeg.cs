using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using MozJpegSharp;
using ImageMagick;

namespace Cupscale.ImageUtils
{
    class MozJpeg
    {
        public static Bitmap ReadImage(string path)
        {
            return (Bitmap)ImgUtils.GetImage(path);
        }

        public static Bitmap ReadImage(MagickImage image)
        {
            using (MemoryStream memStream = new MemoryStream())
            {
                image.Write(memStream);

                return new Bitmap(memStream);
            }
        }

        public static void Encode(MagickImage image, string outPath, int q, bool chromaSubSample = true)
        {
            Bitmap bitmap_image = ReadImage(image);
            Encode_Bitmap(bitmap_image, outPath, q, chromaSubSample);
        }

        public static void Encode(string path, string outpath, int q, bool chromaSubSample = true)
        {
            Bitmap bitmap_image = ReadImage(path);
            Encode_Bitmap(bitmap_image, outpath, q, chromaSubSample);
        }

        public static void Encode_Bitmap(Bitmap bmp, string outPath, int q, bool chromaSubSample = true)
        {
            try
            {
                var commpressor = new TJCompressor();
                byte[] compressed;
                TJSubsamplingOption subSample = TJSubsamplingOption.Chrominance420;
                if (!chromaSubSample)
                    subSample = TJSubsamplingOption.Chrominance444;
                compressed = commpressor.Compress(bmp, subSample, q, TJFlags.None);
                File.WriteAllBytes(outPath, compressed);
                Logger.Log("[MozJpeg] Written image to " + outPath);
            }
            catch (TypeInitializationException e)
            {
                Logger.ErrorMessage($"MozJpeg Initialization Error: {e.InnerException.Message}\n", e);
            }
            catch (Exception e)
            {
                Logger.ErrorMessage("MozJpeg Error: ", e);
            }
        }
    }
}
