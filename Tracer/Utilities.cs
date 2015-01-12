using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ManagedCuda.VectorTypes;
using Tracer.Classes;
using Tracer.Classes.Util;
using Tracer.TracerEventArgs;

namespace Tracer
{
    public static class Utilities
    {
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

        public static Bitmap ConvertFloat3Array( uint Samples, int Width, int Height, float3 [ ] Array )
        {
            byte [ ] ByteArray = new byte[ Array.Length * 4 ];

            // Fill the byte array in parallel, to speed it up.
            Parallel.For( 0, Array.Length, Var =>
            {
                float3 Val = Array[ Var ] / Samples;
                int X = Var % Width;
                int Y = ( Var - X ) / Width;
                //Console.WriteLine( "{0}, {1}: {2}", X, Y, Val );
                Var *= 4;
                ByteArray[ Var ] = ( byte ) MathHelper.Clamp( Val.z * 255, 0, 255 );
                ByteArray[ Var + 1 ] = ( byte ) MathHelper.Clamp( Val.y * 255, 0, 255 );
                ByteArray[ Var + 2 ] = ( byte ) MathHelper.Clamp( Val.x * 255, 0, 255 );
                ByteArray[ Var + 3 ] = 255;
            } );

            return ConvertByteArray( Width, Height, ByteArray );
        }

        public static Bitmap ConvertSample( RenderSampleEventArgs Args )
        {
            byte [ ] ByteArray = new byte[ Args.Data.Length * 4 ];

            // Fill the byte array in parallel, to speed it up.
            Parallel.For( 0, Args.Data.Length, Var =>
            {
                float3 Val = Args.Data[ Var ];
                int X = Var % Args.Width;
                int Y = ( Var - X ) / Args.Width;
                if ( X >= Args.StartX &&
                     Y >= Args.StartY &&
                     X < Args.EndX &&
                     Y < Args.EndY )
                    Val /= Args.AreaSampleCount;
                else
                    Val /= Args.TotalAreaSamples;

                //Console.WriteLine( "{0}, {1}: {2}", X, Y, Val );
                Var *= 4;
                ByteArray[ Var ] = ( byte ) MathHelper.Clamp( Val.z * 255, 0, 255 );
                ByteArray[ Var + 1 ] = ( byte ) MathHelper.Clamp( Val.y * 255, 0, 255 );
                ByteArray[ Var + 2 ] = ( byte ) MathHelper.Clamp( Val.x * 255, 0, 255 );
                ByteArray[ Var + 3 ] = 255;
            } );

            return ConvertByteArray( Args.Width, Args.Height, ByteArray );
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
