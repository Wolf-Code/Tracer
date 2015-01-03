using System;

namespace Tracer
{
    public class RendererProgressEventArgs
    {
        public float Progress;
        public float TotalProgress;
        public TimeSpan ProgressTime;
        public TimeSpan AverageProgressTime;
    }
}