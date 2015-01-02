using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ManagedCuda.VectorTypes;
using Tracer.Classes;
using Tracer.Classes.Util;
using Color = System.Drawing.Color;

namespace Tracer
{
    public static class Utilities
    {
        public static Bitmap ConvertByteArray( int Width, int Height, byte [ ] Data )
        {
            //Here create the Bitmap to the know height, width and format
            Bitmap bmp = new Bitmap( Width, Height, PixelFormat.Format32bppArgb );
            //Bitmap bmp = new Bitmap(width, height);

            //Create a BitmapData and Lock all pixels to be written 
            BitmapData bmpData = bmp.LockBits(
                new Rectangle( 0, 0, bmp.Width, bmp.Height ),
                ImageLockMode.WriteOnly, bmp.PixelFormat );

            //Copy the data from the byte array into BitmapData.Scan0
            Marshal.Copy( Data, 0, bmpData.Scan0, Data.Length );
            //Unlock the pixels
            bmp.UnlockBits( bmpData );

            return bmp;
        }

        public static Bitmap ConvertFloat3Array( uint Samples, int Width, int Height, float3 [ ] Array )
        {
            int WH = Width * Height;
            byte [ ] ByteArray = new byte[ WH * 4 ];

            Parallel.For( 0, WH, Var =>
            {
                float3 Val = Array[ Var ] / Samples;
                Var *= 4;
                ByteArray[ Var ] = ( byte ) MathHelper.Clamp( ( int ) ( Val.z * 255 ), 0, 255 );
                ByteArray[ Var + 1 ] = ( byte ) MathHelper.Clamp( ( int ) ( Val.y * 255 ), 0, 255 );
                ByteArray[ Var + 2 ] = ( byte ) MathHelper.Clamp( ( int ) ( Val.x * 255 ), 0, 255 );
                ByteArray[ Var + 3 ] = 255;
            } );

            return ConvertByteArray( Width, Height, ByteArray );
        }

        public static Classes.Util.Color VectorToColor( Vector3 V )
        {
            return new Classes.Util.Color( V.X, V.Y, V.Z );
        }

        public static Vector3 ColorToVector( Classes.Util.Color C )
        {
            return new Vector3( C.R, C.G, C.B );
        }
    }
}
