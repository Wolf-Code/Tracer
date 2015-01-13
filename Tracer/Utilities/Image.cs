using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ManagedCuda.VectorTypes;

namespace Tracer.Utilities
{
    public static class Image
    {
        public static byte [ ] GetBitmapBytes( Bitmap Image )
        {
            BitmapData bmpData = Image.LockBits(
                new Rectangle( 0, 0, Image.Width, Image.Height ),
                ImageLockMode.ReadOnly, Image.PixelFormat );

            byte [ ] Data = new byte[ bmpData.Stride * Image.Height ];

            Marshal.Copy( bmpData.Scan0, Data, 0, Data.Length );

            Image.UnlockBits( bmpData );

            return Data;
        }

        public static Bitmap ConvertByteArray( int Width, int Height, byte [ ] Data )
        {
            Bitmap Img = new Bitmap( Width, Height, PixelFormat.Format32bppArgb );

            // Lock bitmap data
            BitmapData bmpData = Img.LockBits(
                new Rectangle( 0, 0, Img.Width, Img.Height ),
                ImageLockMode.WriteOnly, Img.PixelFormat );

            // Copy my data into the bitmap
            Marshal.Copy( Data, 0, bmpData.Scan0, Data.Length );

            // Unlock bitmap data
            Img.UnlockBits( bmpData );

            return Img;
        }

        public static Bitmap FillBitmapArea( Bitmap Image, uint Samples, int StartX, int StartY, int Width, int Height,
            float3 [ ] Array )
        {
            byte [ ] ByteArray = GetBitmapBytes( Image );

            int ImgW = Image.Width;
            int ImgH = Image.Height;

            // Fill the byte array in parallel, to speed it up.
            Parallel.For( 0, Array.Length, Var =>
            {
                float3 Val = Array[ Var ] / Samples;
                int X = Var % Width;
                int Y = ( Var - X ) / Width;

                int RealX = X + StartX;
                int RealY = Y + StartY;

                int ID = RealY * ImgW + RealX;
                ID *= 4;
                //Console.WriteLine( "{0}, {1}: {2}", X, Y, Val );

                ByteArray[ ID ] = ( byte ) MathHelper.Clamp( Val.z * 255, 0, 255 );
                ByteArray[ ID + 1 ] = ( byte ) MathHelper.Clamp( Val.y * 255, 0, 255 );
                ByteArray[ ID + 2 ] = ( byte ) MathHelper.Clamp( Val.x * 255, 0, 255 );
                ByteArray[ ID + 3 ] = 255;
            } );

            return ConvertByteArray( ImgW, ImgH, ByteArray );
        }
    }
}