using System;
using System.Diagnostics;
using System.Drawing;
using System.Threading;
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

        private static CUDAObject [ ] Objects
        {
            get
            {
                Scene Scn = new Scene( );

                Sphere Light = Scn.AddSphere( new Vector3( 0, 2000 + 90 - .15f, 0 ), 2000 );
                Light.Material.Radiance = new Classes.Util.Color( 1f, 1f, 1f );

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

                Sphere MirrorSphere = Scn.AddSphere( new Vector3( 20, 40, -20 ), 20 );
                MirrorSphere.Material.Type = CUDAMaterialType.Reflective;
                MirrorSphere.Material.Glossyness = .5f;

                return Scn.ToCUDA( );
            }
        }

        private static CudaKernel RenderKernel;
        private const int ThreadsPerBlock = 32;

        private static void InitKernels( )
        {
            CudaContext cntxt = new CudaContext( );
            CUmodule cumodule = cntxt.LoadModule( Path );
            RenderKernel = new CudaKernel( "TraceKernelRegion", cumodule, cntxt );
            RenderKernel.BlockDimensions = new dim3( ThreadsPerBlock, ThreadsPerBlock );
        }

        private static float3 [ ] RenderImage( )
        {
            int W = ( int ) Renderer.Cam.Resolution.X;
            int H = ( int ) Renderer.Cam.Resolution.Y;
            int WH = W * H;
            // init parameters
            CudaDeviceVariable<float3> vector_hostOut = new CudaDeviceVariable<float3>( WH );
            // run cuda method
            CudaDeviceVariable<float3> vector_Input = new CudaDeviceVariable<float3>( WH );
            float3[ ] In = new float3[ WH ];
            vector_Input.CopyToDevice( In );

            CUDAObject[ ] Objs = Objects;

            CudaDeviceVariable<CUDAObject> Obj = new CudaDeviceVariable<CUDAObject>( Objs.Length );
            Obj.CopyToDevice( Objs );

            RenderKernel.SetConstantVariable( "ObjectArray", Obj.DevicePointer );
            RenderKernel.SetConstantVariable( "Objects", Objs.Length );
            RenderKernel.SetConstantVariable( "Camera", Renderer.Cam.ToCamData( ) );

            int XDivide = 8;
            int YDivide = 8;

            int DivW = W / XDivide;
            int DivH = H / YDivide;

            TimeSpan PrevTimeSpan = new TimeSpan( );
            TimeSpan Average = new TimeSpan( );
            int Areas = 0;
            int TotalAreas = XDivide * YDivide;
            long Seed = DateTime.Now.Second;

            Stopwatch Watch = new Stopwatch( );
            Watch.Start( );
            
            for ( int X = 0; X < XDivide; X++ )
            {
                for ( int Y = 0; Y < YDivide; Y++ )
                {
                    RenderRegion( ref Seed, vector_Input, vector_hostOut, X * DivW, Y * DivH, DivW, DivH );
                    Areas++;

                    TimeSpan S = Watch.Elapsed - PrevTimeSpan;
                    PrevTimeSpan = Watch.Elapsed;
                    Average += S;
                    Seed += DivW * DivH;
                    Console.WriteLine(
                        "Area {0}/{1} took {2}. Expected time until completion: {3}",
                        Areas, TotalAreas, S, new TimeSpan( ( TotalAreas - Areas ) * ( Average.Ticks / Areas ) ) );
                }
            }

            Watch.Stop( );
            Average = new TimeSpan( Average.Ticks / TotalAreas );
            Console.WriteLine( "Render time: {0}. Average area time: {1}", Watch.Elapsed, Average );

            // copy return to host
            float3[ ] output = new float3[ WH ];
            vector_hostOut.CopyToHost( output );

            return output;
        }

        private static void RenderRegion( ref long Seed, CudaDeviceVariable<float3> Input, CudaDeviceVariable<float3> Output, int StartX, int StartY, int W, int H )
        {
            RenderKernel.GridDimensions = new dim3( W / ThreadsPerBlock + 1, H / ThreadsPerBlock + 1 );

            CUdeviceptr InPTR = Input.DevicePointer;
            uint Samples = RayCaster.Samples;
            for ( int Q = 0; Q < Samples; Q++ )
            {
                RenderKernel.SetConstantVariable( "Seed", Seed );
                RenderKernel.Run( InPTR, StartX, StartY, StartX + W, StartY + H, Output.DevicePointer );

                Seed += W * H;

                InPTR = Output.DevicePointer;
            }
        }

        private static Func<float3 [ ]> Render = ( ) =>
        {
            // [ Gridx * Blockx, Gridy * Blocky ]
            int W = ( int ) Renderer.Cam.Resolution.X;
            int H = ( int ) Renderer.Cam.Resolution.Y;

            RenderKernel.GridDimensions = new dim3( W / ThreadsPerBlock + 1, H / ThreadsPerBlock + 1 );
            RenderKernel.BlockDimensions = new dim3( ThreadsPerBlock, ThreadsPerBlock );

            CudaDeviceVariable<CUDAObject> Obj = new CudaDeviceVariable<CUDAObject>( Objects.Length );
            Obj.CopyToDevice( Objects );

            RenderKernel.SetConstantVariable( "ObjectArray", Obj.DevicePointer );
            RenderKernel.SetConstantVariable( "Objects", Objects.Length );
            RenderKernel.SetConstantVariable( "Camera", Renderer.Cam.ToCamData( ) );

            // init parameters
            CudaDeviceVariable<float3> vector_hostOut = new CudaDeviceVariable<float3>( W * H );
            // run cuda method
            CudaDeviceVariable<float3> vector_Input = new CudaDeviceVariable<float3>( W * H );
            float3 [ ] In = new float3[ W * H ];
            vector_Input.CopyToDevice( In );

            Stopwatch Watch = new Stopwatch( );
            Watch.Start( );
            int Samples = ( int ) RayCaster.Samples;
            TimeSpan PrevTimeSpan = new TimeSpan( );
            TimeSpan Average = new TimeSpan( );
            CUdeviceptr InPTR = vector_Input.DevicePointer;

            long Seed = DateTime.Now.Second;

            for ( int Q = 0; Q < Samples; Q++ )
            {
                RenderKernel.SetConstantVariable( "Seed", Seed );
                RenderKernel.Run( InPTR, vector_hostOut.DevicePointer );

                InPTR = vector_hostOut.DevicePointer;
                TimeSpan S = Watch.Elapsed - PrevTimeSpan;
                PrevTimeSpan = Watch.Elapsed;
                Average += S;
                Seed += W * H;
                Console.WriteLine(
                    "Sample {0}/{1} took {2}. Expected time until completion: {3}",
                    Q, Samples, S, new TimeSpan( ( Samples - Q ) * S.Ticks ) );
            }
            Watch.Stop( );
            Average = new TimeSpan( Average.Ticks / Samples );
            Console.WriteLine( "Render time: {0}. Average sample time: {1}", Watch.Elapsed, Average );

            // copy return to host
            float3 [ ] output = new float3[ W * H ];
            vector_hostOut.CopyToHost( output );

            return output;
        };

        public static void Run( )
        {
            new Thread( ( ) =>
            {
            InitKernels( );


            int W = ( int ) Renderer.Cam.Resolution.X;
            int H = ( int )Renderer.Cam.Resolution.Y;

                float3 [ ] result = RenderImage( );
            
                Bitmap Bmp = new Bitmap( W, H );
                for ( int Q = 0; Q < result.Length; Q++ )
                {
                    int X = Q % W;
                    int Y = ( Q - X ) / W;
                    float3 F = result[ Q ] / RayCaster.Samples;

                    Color C = Color.FromArgb( 255,
                        MathHelper.Clamp( ( int ) ( F.x * 255 ), 0, 255 ),
                        MathHelper.Clamp( ( int ) ( F.y * 255 ), 0, 255 ),
                        MathHelper.Clamp( ( int ) ( F.z * 255 ), 0, 255 ) );
                    Bmp.SetPixel( X, Y, C );
                    //Console.WriteLine( "{0}, {1}: {2}", X, Y, F );
                }

                Bmp.Save( @"D:\Temp\testimage.png" );
                Process.Start( @"D:\Temp\testimage.png" );
            } ).Start( );
        }
    }
}