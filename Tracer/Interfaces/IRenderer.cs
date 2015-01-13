using System;
using System.Collections.Generic;
using Tracer.Classes.SceneObjects;
using Tracer.TracerEventArgs;

namespace Tracer.Interfaces
{
    internal interface IRenderer
    {
        event EventHandler<RendererFinishedEventArgs> OnFinished;
        event EventHandler<RenderSampleEventArgs> OnSampleFinished;

        void RenderImage( uint AreaDivider, Scene Scn, uint Samples, uint Depth );
        void Cancel( );
        void NextArea( );
        void Run( );

        List<IDevice> GetDevices( );
    }
}