using System.Runtime.InteropServices;
using ManagedCuda.VectorTypes;

namespace Tracer.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    public struct CUDAPlaneObject
    {
        public float3 Normal;
        public float Offset;
    }
}
