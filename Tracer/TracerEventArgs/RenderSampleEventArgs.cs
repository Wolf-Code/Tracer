using System;
using System.Drawing;
using ManagedCuda.VectorTypes;

namespace Tracer.TracerEventArgs
{
    public class RenderSampleEventArgs
    {
        public float3 [ ] Data;
        public Bitmap Image;
        public int AreaSampleCount;
        public int TotalAreaSamples;
        public int StartX;
        public int StartY;
        public int EndX;
        public int EndY;
        public int Width;
        public int Height;

        public TimeSpan AverageSampleTime;
        public TimeSpan Time;
        public float Progress;
        public int TotalSamples;
    }
}