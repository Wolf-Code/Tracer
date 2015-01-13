using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using Tracer.Classes.Objects;
using Tracer.CUDA;
using Tracer.Interfaces;
using Tracer.Properties;
using Tracer.TracerEventArgs;
using SceneCUDAData = System.Tuple<Tracer.CUDA.CUDAObject[], uint[]>;

namespace Tracer.Renderers
{
    public class CUDARenderer : IRenderer
    {
        public event EventHandler<RendererFinishedEventArgs> OnFinished;
        public event EventHandler<RenderSampleEventArgs> OnSampleFinished; 

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
        private Bitmap LastImage;
        private Stopwatch Watch;
        private uint TotalSamples;
        private uint Samples;
        private TimeSpan Average;
        private DateTime Start;
        private bool SkipToNextArea;

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
            try
            {
                CudaLinker linker = new CudaLinker( options );
                linker.AddFile( @"kernel.ptx", CUJITInputType.PTX, null );
                linker.AddFile( @"Material.ptx", CUJITInputType.PTX, null );
                linker.AddFile( @"VectorMath.ptx", CUJITInputType.PTX, null );

                //important: add the device runtime library!
                linker.AddFile( @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\lib\Win32\cudadevrt.lib",
                    CUJITInputType.Library, null );
                byte [ ] tempArray = linker.Complete( );
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
            LastImage = new Bitmap( W, H );
            this.Start = DateTime.Now;

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

            this.Samples = 0;
            this.TotalSamples = ( XDivide ) * ( YDivide ) * Samples;
            long Seed = RNG.Next( 0, Int32.MaxValue );
            float3[ ] output = new float3[ WH ];
            this.Watch = new Stopwatch( );

            // init parameters
            using ( CudaDeviceVariable<float3> CUDAVar_Output = new CudaDeviceVariable<float3>( WH ) )
            {
                // run cuda method
                using ( CudaDeviceVariable<float3> CUDAVar_Input = new CudaDeviceVariable<float3>( WH ) )
                {
                    float3 [ ] In = new float3[ WH ];
                    CUDAVar_Input.CopyToDevice( In );
                    CUDAVar_Output.CopyToDevice( In );

                    for ( int X = 0; X < XDivide; X++ )
                    {
                        if ( CancelThread )
                            break;

                        for ( int Y = 0; Y < YDivide; Y++ )
                        {
                            if ( CancelThread )
                                break;

                            RenderRegion( Samples, ref Seed, CUDAVar_Input, CUDAVar_Output, X * DivW, Y * DivH, DivW, DivH );
                            Seed += DivW * DivH;
                        }
                    }
                }

                // copy return to host
                CUDAVar_Output.CopyToHost( output );
            }

            Obj.Dispose( );
            Lights.Dispose( );

            if ( OnFinished != null )
                OnFinished.Invoke( null, new RendererFinishedEventArgs
                {
                    AverageProgressTime = new TimeSpan((DateTime.Now-Start).Ticks/this.TotalSamples),
                    Time = DateTime.Now-Start,
                    Image = LastImage
                } );

            if ( CancelThread )
                CancelThread = false;
        }

        private void RenderRegion( uint Samples, ref long Seed, CudaDeviceVariable<float3> Input,
            CudaDeviceVariable<float3> Output, int StartX, int StartY, int W, int H )
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
                if ( CancelThread )
                    return;

                if ( SkipToNextArea )
                {
                    this.Samples += ( uint ) ( Samples - Q );
                    SkipToNextArea = false;
                    return;
                }

                this.Watch.Restart( );

                RenderKernel.SetConstantVariable( "Seed", Seed );
                RenderKernel.Run( InPTR, StartX, StartY, EndX, EndY, Output.DevicePointer );


                this.Samples++;
                Seed += W * H;

                float3 [ ] Data = new float3[ ( int ) ( Scn.Camera.Resolution.X * Scn.Camera.Resolution.Y ) ];
                Output.CopyToHost( Data );

                this.Watch.Stop( );

                TimeSpan RenderSampleTime = Watch.Elapsed;
                Average = new TimeSpan( ( DateTime.Now - this.Start ).Ticks / this.Samples );

                RenderSampleEventArgs E = new RenderSampleEventArgs
                {
                    Data = Data,
                    AreaSampleCount = Q + 1,
                    TotalAreaSamples = ( int ) Samples,
                    StartX = StartX,
                    StartY = StartY,
                    EndX = EndX,
                    EndY = EndY,
                    Width = ( int ) Scn.Camera.Resolution.X,
                    Height = ( int ) Scn.Camera.Resolution.Y,

                    AverageSampleTime = Average,
                    TotalSamples = ( int ) this.TotalSamples,
                    Progress = ( float ) this.Samples / this.TotalSamples,
                    Time = RenderSampleTime
                };


                LastImage = Utilities.ConvertSample( E );
                E.Image = LastImage;

                if ( OnSampleFinished != null )
                    OnSampleFinished.Invoke( this, E );

                InPTR = Output.DevicePointer;
            }
        }

        public void Cancel( )
        {
            //RenderThread.Abort( );
            CancelThread = true;
        }

        public void NextArea( )
        {
            this.SkipToNextArea = true;
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