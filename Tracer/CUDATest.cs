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
using Tracer.Properties;
using Color = System.Drawing.Color;

namespace Tracer
{
    public class CUDAProgressEventArgs
    {
        public float Progress;
        public float TotalProgress;
        public TimeSpan ProgressTime;
        public TimeSpan AverageProgressTime;
    }

    public class CUDAFinishedEventArgs
    {
        public TimeSpan Time;
        public TimeSpan AverageProgressTime;
        public Bitmap Image;
    }

    public static class CUDATest
    {
        public static event EventHandler<CUDAProgressEventArgs> OnProgress;
        public static event EventHandler<CUDAFinishedEventArgs> OnFinished;
        private static Thread RenderThread;

        public static string Path
        {
            get { return Environment.CurrentDirectory + "\\kernel.ptx"; }
        }

        private static Scene DefaultScene
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
                MirrorSphere.Material.Glossyness = .1f;

                return Scn;
            }
        }

        private static CudaKernel RenderKernel;
        private const int ThreadsPerBlock = 32;

        private static void InitKernels( )
        {
            CudaContext cntxt = new CudaContext( );
            CUmodule cumodule = cntxt.LoadModule( Path );
            RenderKernel = new CudaKernel( "TraceKernelRegion", cumodule, cntxt )
            {
                BlockDimensions = new dim3( ThreadsPerBlock, ThreadsPerBlock )
            };
        }

        private static void RenderImage( Scene Scn )
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

            CUDAObject [ ] Objs = Scn.ToCUDA( );

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

                    if ( OnProgress != null )
                        OnProgress.Invoke( null, new CUDAProgressEventArgs
                        {
                            Progress = ( 1f / TotalAreas ),
                            ProgressTime = S,
                            TotalProgress = ( Areas / ( float )TotalAreas ),
                            AverageProgressTime = new TimeSpan( Average.Ticks / Areas )
                        } );
                }
            }

            Watch.Stop( );
            Average = new TimeSpan( Average.Ticks / TotalAreas );

            // copy return to host
            float3[ ] output = new float3[ WH ];
            vector_hostOut.CopyToHost( output );

            Bitmap Bmp = new Bitmap( W, H );
            for ( int Q = 0; Q < output.Length; Q++ )
            {
                int X = Q % W;
                int Y = ( Q - X ) / W;
                float3 F = output[ Q ] / Settings.Default.Render_Samples;

                Color C = Color.FromArgb( 255,
                    MathHelper.Clamp( ( int )( F.x * 255 ), 0, 255 ),
                    MathHelper.Clamp( ( int )( F.y * 255 ), 0, 255 ),
                    MathHelper.Clamp( ( int )( F.z * 255 ), 0, 255 ) );

                Bmp.SetPixel( X, Y, C );
            }

            if ( OnFinished != null )
                OnFinished.Invoke( null, new CUDAFinishedEventArgs
                {
                    AverageProgressTime = Average,
                    Time = Watch.Elapsed,
                    Image = Bmp
                } );
        }

        private static void RenderRegion( ref long Seed, CudaDeviceVariable<float3> Input, CudaDeviceVariable<float3> Output, int StartX, int StartY, int W, int H )
        {
            RenderKernel.GridDimensions = new dim3( W / ThreadsPerBlock + 1, H / ThreadsPerBlock + 1 );

            CUdeviceptr InPTR = Input.DevicePointer;
            uint Samples = Settings.Default.Render_Samples;
            for ( int Q = 0; Q < Samples; Q++ )
            {
                RenderKernel.SetConstantVariable( "Seed", Seed );
                RenderKernel.Run( InPTR, StartX, StartY, StartX + W, StartY + H, Output.DevicePointer );

                Seed += W * H;

                InPTR = Output.DevicePointer;
            }
        }

        public static void Cancel( )
        {
            RenderThread.Abort( );
        }

        public static void Run( )
        {
            RenderThread = new Thread( ( ) =>
            {
                InitKernels( );
                RenderImage( DefaultScene );
            } );

            RenderThread.Start( );
        }
    }
}