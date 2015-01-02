using System;
using System.Diagnostics;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
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

    public static class CUDAInterface
    {
        public static event EventHandler<CUDAProgressEventArgs> OnProgress;
        public static event EventHandler<CUDAFinishedEventArgs> OnFinished;
        private static Thread RenderThread;

        public static string Path
        {
            get { return Environment.CurrentDirectory + "\\kernel.ptx"; }
        }

        public static Scene DefaultScene
        {
            get
            {
                Scene Scn = new Scene
                {
                    Camera =
                    {
                        Position = new Vector3( 0, 45, 80 ),
                        Angle = new Angle { Pitch = 0, Yaw = 0, Roll = 0f }
                    }
                };


                Sphere Light = Scn.AddSphere( new Vector3( 0, 2000 + 90 - .15f, 0 ), 2000 );
                Light.Name = "Light";
                Light.Material.Radiance = new Classes.Util.Color( 12f, 12f, 12f );

                Plane Floor = Scn.AddPlane( new Vector3( 0, 1, 0 ), 0 );
                Floor.Name = "Floor";
                Floor.Material.Color = new Classes.Util.Color( 1f, 1f, 1f );

                Plane Front = Scn.AddPlane( new Vector3( 0, 0, 1 ), 90 );
                Front.Name = "Front";
                Front.Material.Color = new Classes.Util.Color( 1f, 1f, 1f );

                Plane Back = Scn.AddPlane( new Vector3( 0, 0, -1 ), 90 );
                Back.Name = "Back";
                Back.Material.Color = new Classes.Util.Color( 1f, 1f, 1f );

                Plane Ceiling = Scn.AddPlane( new Vector3( 0, -1, 0 ), 90 );
                Ceiling.Name = "Ceiling";
                Ceiling.Material.Color = new Classes.Util.Color( 1f, 1f, 1f );

                Plane Left = Scn.AddPlane( new Vector3( 1, 0, 0 ), 90 );
                Left.Name = "Left";
                Left.Material.Color = new Classes.Util.Color( 1f, 0f, 0f );

                Plane Right = Scn.AddPlane( new Vector3( -1, 0, 0 ), 90 );
                Right.Name = "Right";
                Right.Material.Color = new Classes.Util.Color( 0, 0f, 1f );

                Sphere GreenSphere = Scn.AddSphere( new Vector3( -20, 50, -30 ), 20 );
                GreenSphere.Name = "Green sphere";
                GreenSphere.Material.Color = new Classes.Util.Color( 0, 1f, 0f );

                Sphere MirrorSphere = Scn.AddSphere( new Vector3( 20, 40, -20 ), 20 );
                MirrorSphere.Name = "Mirror sphere";
                MirrorSphere.Material.Type = CUDAMaterialType.Reflective;
                MirrorSphere.Material.Glossyness = 0f;

                return Scn;
            }
        }

        private static CudaKernel RenderKernel;
        private const int ThreadsPerBlock = 32;
        private static bool CancelThread;

        private static void InitKernels( )
        {
            CudaContext cntxt = new CudaContext( );
            CUmodule cumodule = cntxt.LoadModule( Path );
            RenderKernel = new CudaKernel( "TraceKernelRegion", cumodule, cntxt )
            {
                BlockDimensions = new dim3( ThreadsPerBlock, ThreadsPerBlock )
            };
        }

        private static void RenderImage( Scene Scn, uint Samples, uint Depth )
        {
            int W = ( int ) Scn.Camera.Resolution.X;
            int H = ( int ) Scn.Camera.Resolution.Y;
            int WH = W * H;

            CUDAObject [ ] Objs = Scn.ToCUDA( );

            CudaDeviceVariable<CUDAObject> Obj = new CudaDeviceVariable<CUDAObject>( Objs.Length );
            Obj.CopyToDevice( Objs );

            RenderKernel.SetConstantVariable( "ObjectArray", Obj.DevicePointer );
            RenderKernel.SetConstantVariable( "Objects", Objs.Length );
            RenderKernel.SetConstantVariable( "Camera", Scn.Camera.ToCamData( ) );

            int XDivide = 8;
            int YDivide = 8;

            int DivW = W / XDivide;
            int DivH = H / YDivide;

            TimeSpan PrevTimeSpan = new TimeSpan( );
            TimeSpan Average = new TimeSpan( );
            int Areas = 0;
            int TotalAreas = XDivide * YDivide;
            long Seed = DateTime.Now.Second;
            float3[ ] output = new float3[ WH ];
            Stopwatch Watch = new Stopwatch( );

            // init parameters
            using ( CudaDeviceVariable<float3> CUDAVar_Output = new CudaDeviceVariable<float3>( WH ) )
            {
                // run cuda method
                using ( CudaDeviceVariable<float3> CUDAVar_Input = new CudaDeviceVariable<float3>( WH ) )
                {
                    float3 [ ] In = new float3[ WH ];
                    CUDAVar_Input.CopyToDevice( In );

                    Watch.Start( );

                    for ( int X = 0; X < XDivide; X++ )
                    {
                        if ( CancelThread )
                            break;

                        for ( int Y = 0; Y < YDivide; Y++ )
                        {
                            if ( CancelThread )
                                break;

                            RenderRegion( Samples, ref Seed, CUDAVar_Input, CUDAVar_Output, X * DivW, Y * DivH, DivW,
                                DivH );
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
                                    TotalProgress = ( Areas / ( float ) TotalAreas ),
                                    AverageProgressTime = new TimeSpan( Average.Ticks / Areas )
                                } );
                        }
                    }

                    Watch.Stop( );
                }
                Average = new TimeSpan( Average.Ticks / TotalAreas );

                // copy return to host
                CUDAVar_Output.CopyToHost( output );
            }

            if ( OnFinished != null )
                OnFinished.Invoke( null, new CUDAFinishedEventArgs
                {
                    AverageProgressTime = Average,
                    Time = Watch.Elapsed,
                    Image = Utilities.ConvertFloat3Array( Samples, W, H, output )
                } );

            if ( CancelThread )
                CancelThread = false;
        }

        private static void RenderRegion( uint Samples, ref long Seed, CudaDeviceVariable<float3> Input, CudaDeviceVariable<float3> Output, int StartX, int StartY, int W, int H )
        {
            RenderKernel.GridDimensions = new dim3( W / ThreadsPerBlock + 1, H / ThreadsPerBlock + 1 );

            CUdeviceptr InPTR = Input.DevicePointer;
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
            //RenderThread.Abort( );
            CancelThread = true;
        }

        public static void Run( )
        {
            RenderThread = new Thread( ( ) =>
            {
                InitKernels( );
                RenderImage( Renderer.Scene, Settings.Default.Render_Samples, Settings.Default.Render_MaxDepth );
            } );

            RenderThread.Start( );
        }
    }
}