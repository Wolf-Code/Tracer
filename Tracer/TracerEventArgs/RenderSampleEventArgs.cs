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

        public TimeSpan AverageSampleTime;
        public TimeSpan Time;
        public float Progress;
        public int TotalSamples;
    }
}