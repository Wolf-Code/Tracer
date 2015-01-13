using System.Runtime.InteropServices;
using ManagedCuda.VectorTypes;

namespace Tracer.Structs.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    public struct CUDACamData
    {
        public float3 Position;
        public float3 Forward;
        public float3 Right;
        public float3 Up;
        public float A;
        public float Width;
        public float Height;
    }
}