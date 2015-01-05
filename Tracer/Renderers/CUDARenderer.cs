using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using Tracer.Classes.Objects;
using Tracer.CUDA;
using Tracer.Interfaces;
using Tracer.Properties;
using SceneCUDAData = System.Tuple<Tracer.CUDA.CUDAObject[], Tracer.CUDA.CUDAObject[]>;

namespace Tracer.Renderers
{
    public class CUDARenderer : IRenderer
    {
        public event EventHandler<RendererProgressEventArgs> OnProgress;
        public event EventHandler<RendererFinishedEventArgs> OnFinished;
        private Thread RenderThread;

        public static string Path
        {
            get { return Environment.CurrentDirectory + "\\kernel.ptx"; }
        }

        private static Random RNG = new Random( );

        private CudaKernel RenderKernel;
        private const int ThreadsPerBlock = 32;
        private bool CancelThread;

        private void InitKernels( )
        {
            CudaContext cntxt = new CudaContext( );
            CUmodule cumodule = cntxt.LoadModule( Path );
            RenderKernel = new CudaKernel( "TraceKernelRegion", cumodule, cntxt )
            {
                BlockDimensions = new dim3( ThreadsPerBlock, ThreadsPerBlock )
            };
        }

        public void RenderImage( Scene Scn, uint Samples, uint Depth )
        {
            int W = ( int ) Scn.Camera.Resolution.X;
            int H = ( int ) Scn.Camera.Resolution.Y;
            int WH = W * H;

            // Item1 = objects, Item2 = lights
            SceneCUDAData Objs = Scn.ToCUDA( );

            CudaDeviceVariable<CUDAObject> Obj = new CudaDeviceVariable<CUDAObject>( Objs.Item1.Length );
            Obj.CopyToDevice( Objs.Item1 );

            CudaDeviceVariable<CUDAObject> Lights = new CudaDeviceVariable<CUDAObject>( Objs.Item2.Length );
            Obj.CopyToDevice( Objs.Item2 );

            foreach ( CUDAObject O in Objs.Item1 )
                Console.WriteLine( "Object {0}", O.ID );

            foreach ( CUDAObject O in Objs.Item2 )
                Console.WriteLine( "Light {0}", O.ID );

            RenderKernel.SetConstantVariable( "ObjectArray", Obj.DevicePointer );
            RenderKernel.SetConstantVariable( "Objects", ( uint )Objs.Item1.Length );
            //RenderKernel.SetConstantVariable( "Lights", Lights.DevicePointer );
            //RenderKernel.SetConstantVariable( "LightCount", ( uint )Objs.Item2.Length );
            RenderKernel.SetConstantVariable( "Camera", Scn.Camera.ToCamData( ) );
            RenderKernel.SetConstantVariable( "MaxDepth", Depth );

            int XDivide = 8;
            int YDivide = 8;

            int DivW = W / XDivide;
            int DivH = H / YDivide;

            TimeSpan PrevTimeSpan = new TimeSpan( );
            TimeSpan Average = new TimeSpan( );
            int Areas = 0;
            int TotalAreas = XDivide * YDivide;
            long Seed = RNG.Next( 0, Int32.MaxValue );
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
                                OnProgress.Invoke( null, new RendererProgressEventArgs
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

            Obj.Dispose( );
            Lights.Dispose( );

            if ( OnFinished != null )
                OnFinished.Invoke( null, new RendererFinishedEventArgs
                {
                    AverageProgressTime = Average,
                    Time = Watch.Elapsed,
                    Image = Utilities.ConvertFloat3Array( Samples, W, H, output )
                } );

            if ( CancelThread )
                CancelThread = false;
        }

        private void RenderRegion( uint Samples, ref long Seed, CudaDeviceVariable<float3> Input, CudaDeviceVariable<float3> Output, int StartX, int StartY, int W, int H )
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

        public void Cancel( )
        {
            //RenderThread.Abort( );
            CancelThread = true;
        }

        public void Run( )
        {
            RenderThread = new Thread( ( ) =>
            {
                InitKernels( );
                RenderImage( Renderer.Scene, Settings.Default.Render_Samples, Settings.Default.Render_MaxDepth );
            } );

            RenderThread.Start( );
        }

        public List<IDevice> GetDevices( )
        {
            List<IDevice> Devices = new List<IDevice>( );
            for ( int X = 0; X < CudaContext.GetDeviceCount( ); X++ )
            {
                Devices.Add( new CUDADevice { Device = CudaContext.GetDeviceInfo( X ) } );
            }

            return Devices;
        }
    }
}