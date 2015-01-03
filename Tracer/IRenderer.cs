using System;
using System.Collections.Generic;
using System.Drawing;
using Tracer.Classes.Objects;

namespace Tracer
{
    public class RendererProgressEventArgs
    {
        public float Progress;
        public float TotalProgress;
        public TimeSpan ProgressTime;
        public TimeSpan AverageProgressTime;
    }

    public class RendererFinishedEventArgs
    {
        public TimeSpan Time;
        public TimeSpan AverageProgressTime;
        public Bitmap Image;
    }

    interface IRenderer
    {
        event EventHandler<RendererProgressEventArgs> OnProgress;
        event EventHandler<RendererFinishedEventArgs> OnFinished;
 
        void RenderImage( Scene Scn, uint Samples, uint Depth );
        void Cancel( );
        void Run( );

        List<IDevice> GetDevices( );
    }
}
