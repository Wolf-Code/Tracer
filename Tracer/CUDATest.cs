using System;
using System.Diagnostics;
using System.Drawing;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using Tracer.Classes;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;
using Tracer.CUDA;
using Color = System.Drawing.Color;

namespace Tracer
{
    public static class CUDATest
    {
        public static string Path
        {
            get { return Environment.CurrentDirectory + "\\kernel.ptx"; }
        }

        static CudaKernel addTwoVectorWithCuda;

        static void InitKernels( )
        {
            CudaContext cntxt = new CudaContext( );
            CUmodule cumodule = cntxt.LoadModule( Path );
            addTwoVectorWithCuda = new CudaKernel( "TraceKernel", cumodule, cntxt );
        }

        private static Func<float3[ , ]> test = ( ) =>
        {
            int ThreadsPerBlock = 32;
            // [ Gridx * Blockx, Gridy * Blocky ]
            int W = ( int ) Renderer.Cam.Resolution.X;
            int H = ( int ) Renderer.Cam.Resolution.Y;

            addTwoVectorWithCuda.GridDimensions = new dim3( W / ThreadsPerBlock + 1, H / ThreadsPerBlock + 1 );
            addTwoVectorWithCuda.BlockDimensions = new dim3( ThreadsPerBlock, ThreadsPerBlock );

            Scene Scn = new Scene( );

            Sphere Light = Scn.AddSphere( new Vector3( 0, 2000 + 90 - .1f, 0 ), 2000 );
            Light.Material.Radiance = new Classes.Util.Color( 1, 1, 1 );

            Plane Floor = Scn.AddPlane( new Vector3( 0, 1, 0 ), 0 );
            Floor.Material.Color = new Classes.Util.Color( 1f, 1f, 1f );

            Plane Front = Scn.AddPlane( new Vector3( 0, 0, 1 ), 90 );
            Front.Material.Color = new Classes.Util.Color( 1f, 1f, 1f );

            Plane Back = Scn.AddPlane( new Vector3( 0, 0, -1 ), 90 );
            Back.Material.Color = new Classes.Util.Color( 1f, 1f, 1f );

            Plane Ceiling = Scn.AddPlane( new Vector3( 0, -1, 0 ), 90 );
            Ceiling.Material.Color = new Classes.Util.Color( 1f, 1f, 1f );

            Plane Left = Scn.AddPlane( new Vector3( 1, 0, 0 ), 90 );
            Left.Material.Color = new Classes.Util.Color( 1f, 0f, 0f );

            Plane Right = Scn.AddPlane( new Vector3( -1, 0, 0 ), 90 );
            Right.Material.Color = new Classes.Util.Color( 0, 0f, 1f );

            Sphere GreenSphere = Scn.AddSphere( new Vector3( -20, 50, -30 ), 20 );
            GreenSphere.Material.Color = new Classes.Util.Color( 0, 1f, 0f );

            CUDAObject[ ] Objects = Scn.ToCUDA( );
            

            CudaDeviceVariable<CUDAObject> Obj = new CudaDeviceVariable<CUDAObject>( Objects.Length );
            Obj.CopyToDevice( Objects );

            addTwoVectorWithCuda.SetConstantVariable( "ObjectArray", Obj.DevicePointer );
            addTwoVectorWithCuda.SetConstantVariable( "Objects", Objects.Length );
            addTwoVectorWithCuda.SetConstantVariable( "Camera", Renderer.Cam.ToCamData( ) );
            addTwoVectorWithCuda.SetConstantVariable( "Seed", DateTime.Now.Second );

            // init parameters
            CudaDeviceVariable<float3> vector_hostOut = new CudaDeviceVariable<float3>( W * H );
            // run cuda method
            addTwoVectorWithCuda.Run( vector_hostOut.DevicePointer );

            // copy return to host
            float3[ ] output = new float3[ W * H ];
            vector_hostOut.CopyToHost( output );

            float3[ , ] Out = new float3[ W, H ];
            for ( int Q = 0; Q < output.Length; Q++ )
            {
                int X = ( Q % W );
                int Y = ( Q / W );

                Out[ X, Y ] = output[ Q ];
            }

            return Out;
        };

        public static void Run( )
        {
            InitKernels( );
            float3[ , ] result = test( );

            Bitmap Bmp = new Bitmap( result.GetLength( 0 ), result.GetLength( 1 ) );
            for( int X = 0; X < result.GetLength( 0 ); X++ )
                for ( int Y = 0; Y < result.GetLength( 1 ); Y++ )
                {
                    float3 F = result[ X, Y ];
                    float Largest = F.x;
                    if ( F.y > Largest )
                        Largest = F.y;
                    if ( F.z > Largest )
                        Largest = F.z;
                    F /= Largest;

                    Color C = Color.FromArgb( 255,
                        MathHelper.Clamp( ( int ) ( F.x * 255 ), 0, 255 ),
                        MathHelper.Clamp( ( int ) ( F.y * 255 ), 0, 255 ),
                        MathHelper.Clamp( ( int ) ( F.z * 255 ), 0, 255 ) );
                    Bmp.SetPixel( X, Y, C );
                    //Console.WriteLine( "{0}, {1}: {2}", X, Y, F );
                }

            Bmp.Save( @"D:\Temp\testimage.png" );
            Process.Start( @"D:\Temp\testimage.png" );
        }
    }
}
