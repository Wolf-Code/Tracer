using System;
using System.Drawing;

namespace Tracer.TracerEventArgs
{
    public class RendererFinishedEventArgs
    {
        public TimeSpan Time;
        public TimeSpan AverageProgressTime;
        public Bitmap Image;
    }
}