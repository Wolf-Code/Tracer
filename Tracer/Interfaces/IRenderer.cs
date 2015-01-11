using System;
using System.Collections.Generic;
using Tracer.Classes.Objects;

namespace Tracer.Interfaces
{
    interface IRenderer
    {
        event EventHandler<RendererProgressEventArgs> OnProgress;
        event EventHandler<RendererFinishedEventArgs> OnFinished;
 
        void RenderImage( uint AreaDivider, Scene Scn, uint Samples, uint Depth );
        void Cancel( );
        void Run( );

        List<IDevice> GetDevices( );
    }
}
