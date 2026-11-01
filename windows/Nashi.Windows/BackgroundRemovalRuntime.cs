using System.Security.Cryptography;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Nashi.Core;

namespace Nashi.Windows;

/// Local U²-Net-P inference. The model is downloaded only after the user opts in,
/// verified before publication, and never receives a path or network connection.
internal static class BackgroundRemovalRuntime
{
    private const string Url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx";
    private const string Sha256="309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8";
    private static readonly string DirectoryPath=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"Nashi","Models","u2netp-v1");
    internal static readonly string ModelPath=Path.Combine(DirectoryPath,"u2netp.onnx");
    private static readonly object Gate=new();private static InferenceSession? session;private static string? sessionPath;
    internal static bool Ready=>Valid(ModelPath);

    internal static async Task EnsureAsync(IProgress<OperationProgress>? progress=null,CancellationToken token=default)
    {
        if(Ready)return;Directory.CreateDirectory(DirectoryPath);var stage=Path.Combine(DirectoryPath,"download-"+Guid.NewGuid().ToString("N")+".onnx");
        try
        {
            await VerifiedDownload.FetchAsync(new(Url),stage,8*1024*1024,value=>progress?.Report(new(value,"Downloading background removal model…")),token);
            if(!Valid(stage))throw new InvalidDataException("Background removal model checksum mismatch.");
            token.ThrowIfCancellationRequested();if(File.Exists(ModelPath))File.Delete(ModelPath);File.Move(stage,ModelPath);
        }
        finally{if(File.Exists(stage))File.Delete(stage);}
    }
    private static bool Valid(string path)
    {
        if(!File.Exists(path)||new FileInfo(path).Length!=4_574_861)return false;
        using var stream=File.OpenRead(path);return Convert.ToHexString(SHA256.HashData(stream)).Equals(Sha256,StringComparison.OrdinalIgnoreCase);
    }
    private static InferenceSession Session(string path)
    {
        lock(Gate)
        {
            if(session is not null&&sessionPath==path)return session;session?.Dispose();sessionPath=path;
            return session=new InferenceSession(path,new SessionOptions{GraphOptimizationLevel=GraphOptimizationLevel.ORT_ENABLE_ALL,IntraOpNumThreads=Math.Max(1,Environment.ProcessorCount/2)});
        }
    }
    internal static BitmapSource Remove(BitmapSource image,CancellationToken token=default)
    {
        var model=ModelPath;if(!Valid(model))throw new InvalidOperationException("Download the background removal model before using Smart mode.");token.ThrowIfCancellationRequested();
        const int size=320;var scaled=new TransformedBitmap(image,new ScaleTransform((double)size/image.PixelWidth,(double)size/image.PixelHeight));var rgb=new FormatConvertedBitmap(scaled,PixelFormats.Bgra32,null,0);var bytes=new byte[size*size*4];rgb.CopyPixels(bytes,size*4,0);
        var input=new DenseTensor<float>(new[]{1,3,size,size});var mean=new[]{.485f,.456f,.406f};var deviation=new[]{.229f,.224f,.225f};
        for(var y=0;y<size;y++)for(var x=0;x<size;x++){var pixel=(y*size+x)*4;input[0,0,y,x]=(bytes[pixel+2]/255f-mean[0])/deviation[0];input[0,1,y,x]=(bytes[pixel+1]/255f-mean[1])/deviation[1];input[0,2,y,x]=(bytes[pixel]/255f-mean[2])/deviation[2];}
        token.ThrowIfCancellationRequested();var runtime=Session(model);var inputName=runtime.InputMetadata.Keys.Single();using var results=runtime.Run([NamedOnnxValue.CreateFromTensor(inputName,input)]);var output=results.First().AsTensor<float>();
        var minimum=float.MaxValue;var maximum=float.MinValue;for(var i=0;i<size*size;i++){var value=output.GetValue(i);minimum=Math.Min(minimum,value);maximum=Math.Max(maximum,value);}var range=Math.Max(1e-6f,maximum-minimum);var mask=new byte[size*size];for(var i=0;i<mask.Length;i++)mask[i]=(byte)Math.Clamp(Math.Round((output.GetValue(i)-minimum)/range*255),0,255);
        var smallMask=BitmapSource.Create(size,size,96,96,PixelFormats.Gray8,null,mask,size);var fullMask=new FormatConvertedBitmap(new TransformedBitmap(smallMask,new ScaleTransform((double)image.PixelWidth/size,(double)image.PixelHeight/size)),PixelFormats.Gray8,null,0);var alpha=new byte[image.PixelWidth*image.PixelHeight];fullMask.CopyPixels(alpha,image.PixelWidth,0);
        var source=image.Format==PixelFormats.Bgra32?image:new FormatConvertedBitmap(image,PixelFormats.Bgra32,null,0);var stride=image.PixelWidth*4;var pixels=new byte[stride*image.PixelHeight];source.CopyPixels(pixels,stride,0);for(var i=0;i<alpha.Length;i++)pixels[i*4+3]=(byte)(pixels[i*4+3]*alpha[i]/255);token.ThrowIfCancellationRequested();
        var result=BitmapSource.Create(image.PixelWidth,image.PixelHeight,image.DpiX,image.DpiY,PixelFormats.Bgra32,null,pixels,stride);result.Freeze();return result;
    }
}
