using System;
using System.Collections.Generic;
using OpenCL.Net;
using Tracer.Classes.Objects;
using Tracer.Interfaces;

namespace Tracer.Renderers
{
    internal class OpenCLRenderer : IRenderer
    {
        public event EventHandler<RendererProgressEventArgs> OnProgress;
        public event EventHandler<RendererFinishedEventArgs> OnFinished;

        private Device _device;
        private Context _context;

        private void CheckErr( ErrorCode err, string name )
        {
            if ( err != ErrorCode.Success )
            {
                Console.WriteLine( "ERROR: " + name + " (" + err.ToString( ) + ")" );
            }
        }

        private void ContextNotify( string errInfo, byte [ ] data, IntPtr cb, IntPtr userData )
        {
            Console.WriteLine( "OpenCL Notification: " + errInfo );
        }

        private void Setup( )
        {
            ErrorCode error;
            Platform [ ] platforms = Cl.GetPlatformIDs( out error );
            List<Device> devicesList = new List<Device>( );

            CheckErr( error, "Cl.GetPlatformIDs" );

            foreach ( Platform platform in platforms )
            {
                string platformName = Cl.GetPlatformInfo( platform, PlatformInfo.Name, out error ).ToString( );
                Console.WriteLine( "Platform: " + platformName );
                CheckErr( error, "Cl.GetPlatformInfo" );
                //We will be looking only for GPU devices
                foreach ( Device device in Cl.GetDeviceIDs( platform, DeviceType.Gpu, out error ) )
                {
                    CheckErr( error, "Cl.GetDeviceIDs" );
                    Console.WriteLine( "Device: " + Cl.GetDeviceInfo( device, DeviceInfo.Name, out error ) );
                    CheckErr( error, "Cl.GetDeviceInfo" );
                    devicesList.Add( device );
                }
            }

            if ( devicesList.Count <= 0 )
            {
                Console.WriteLine( "No devices found." );
                return;
            }

            _device = devicesList[ 0 ];
            _context
                = Cl.CreateContext( null, 1, new [ ] { _device }, ContextNotify,
                    IntPtr.Zero, out error ); //Second parameter is amount of devices
            CheckErr( error, "Cl.CreateContext" );
        }

        public void Run( )
        {
            Setup(  );
        }

        public void Cancel( )
        {
        }

        public void RenderImage( Scene Scn, uint Samples, uint Depth )
        {
        }

        public List<IDevice> GetDevices( )
        {
            return new List<IDevice>( );
        }
    }
}