using System;
using System.Collections.Generic;
using Tracer.Classes.SceneObjects;
using Tracer.Structs;
using Tracer.TracerEventArgs;

namespace Tracer.Interfaces
{
    internal interface IRenderer
    {
        event EventHandler<RendererFinishedEventArgs> OnFinished;
        event EventHandler<RenderSampleEventArgs> OnSampleFinished;

        void RenderImage( ref RenderSettings RenderSetting, Scene Scn );
        void Cancel( );
        void NextArea( );
        void Run( );

        List<IDevice> GetDevices( );
    }
}