using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using Tracer.Classes.Objects;
using Tracer.CUDA;
using Tracer.Interfaces;
using Tracer.Properties;
using SceneCUDAData = System.Tuple<Tracer.CUDA.CUDAObject[], uint[]>;

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

        private static readonly Random RNG = new Random( );

        private CudaKernel RenderKernel;
        private const int ThreadsPerBlock = 32;
        private bool CancelThread;
        private Scene Scn;

        private void InitKernels( )
        {
            CudaContext cntxt = new CudaContext( );
            //Add an info and error buffer to see what the linker wants to tell us:
            CudaJitOptionCollection options = new CudaJitOptionCollection( );
            CudaJOErrorLogBuffer err = new CudaJOErrorLogBuffer( 1024 );
            CudaJOInfoLogBuffer info = new CudaJOInfoLogBuffer( 1024 );
            options.Add( new CudaJOLogVerbose( true ) );
            options.Add( err );
            options.Add( info );
            byte [ ] tempArray = null;
            try
            {
                CudaLinker linker = new CudaLinker( options );
                linker.AddFile( @"kernel.ptx", CUJITInputType.PTX, null );
                linker.AddFile( @"Material.ptx", CUJITInputType.PTX, null );
                linker.AddFile( @"VectorMath.ptx", CUJITInputType.PTX, null );

                //important: add the device runtime library!
                linker.AddFile( @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\lib\Win32\cudadevrt.lib",
                    CUJITInputType.Library, null );
                tempArray = linker.Complete( );
                //MessageBox.Show( info.Value );

                RenderKernel = cntxt.LoadKernelPTX( tempArray, "TraceKernelRegion" );
                RenderKernel.BlockDimensions = new dim3( ThreadsPerBlock, ThreadsPerBlock );
            }

            catch ( Exception E )
            {
                Console.WriteLine( E.Message );
            }
        }

        public void RenderImage( uint AreaDivider, Scene Scn, uint Samples, uint Depth )
        {
            this.Scn = Scn;
            int W = ( int ) Scn.Camera.Resolution.X;
            int H = ( int ) Scn.Camera.Resolution.Y;
            int WH = W * H;

            // Item1 = objects, Item2 = lights
            SceneCUDAData Objs = Scn.ToCUDA( );

            foreach ( CUDAObject O in Objs.Item1 )
                Console.WriteLine( "Object {0}", O.ID );

            foreach ( uint O in Objs.Item2 )
                Console.WriteLine( "Light {0}", O );

            CudaDeviceVariable<CUDAObject> Obj = new CudaDeviceVariable<CUDAObject>( Objs.Item1.Length );
            Obj.CopyToDevice( Objs.Item1 );

            CudaDeviceVariable<uint> Lights = new CudaDeviceVariable<uint>( Objs.Item2.Length );
            Lights.CopyToDevice( Objs.Item2 );

            RenderKernel.SetConstantVariable( "ObjectArray", Obj.DevicePointer );
            RenderKernel.SetConstantVariable( "Objects", ( uint )Objs.Item1.Length );
            RenderKernel.SetConstantVariable( "Lights", Lights.DevicePointer );
            RenderKernel.SetConstantVariable( "LightCount", ( uint )Objs.Item2.Length );
            RenderKernel.SetConstantVariable( "Camera", Scn.Camera.ToCamData( ) );
            RenderKernel.SetConstantVariable( "MaxDepth", Depth );

            uint XDivide = AreaDivider;
            uint YDivide = AreaDivider;

            int DivW = ( int )( W / XDivide );
            int DivH = ( int )( H / YDivide );

            TimeSpan PrevTimeSpan = new TimeSpan( );
            TimeSpan Average = new TimeSpan( );
            int Areas = 0;
            uint TotalAreas = ( XDivide + 1 ) * ( YDivide + 1 );
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

                    for ( int X = 0; X < XDivide + 1; X++ )
                    {
                        if ( CancelThread )
                            break;

                        for ( int Y = 0; Y < YDivide + 1; Y++ )
                        {
                            if ( CancelThread )
                                break;

                            RenderRegion( Samples, ref Seed, CUDAVar_Input, CUDAVar_Output, X * DivW, Y * DivH, DivW, DivH );
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
            if ( StartX >= Scn.Camera.Resolution.X || StartY >= Scn.Camera.Resolution.Y )
                return;

            int EndX = StartX + W;
            int EndY = StartY + H;

            if ( EndX > Scn.Camera.Resolution.X )
                EndX = ( int ) Scn.Camera.Resolution.X;

            if ( EndY > Scn.Camera.Resolution.Y )
                EndY = ( int ) Scn.Camera.Resolution.Y;

            RenderKernel.GridDimensions = new dim3( W / ThreadsPerBlock + 1, H / ThreadsPerBlock + 1 );

            CUdeviceptr InPTR = Input.DevicePointer;
            for ( int Q = 0; Q < Samples; Q++ )
            {
                RenderKernel.SetConstantVariable( "Seed", Seed );
                RenderKernel.Run( InPTR, StartX, StartY, EndX, EndY, Output.DevicePointer );

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
                RenderImage( Settings.Default.Render_AreaDivider, Renderer.Scene, Settings.Default.Render_Samples, Settings.Default.Render_MaxDepth );
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