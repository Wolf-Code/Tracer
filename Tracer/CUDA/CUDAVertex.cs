using System.Runtime.InteropServices;
using ManagedCuda.VectorTypes;

namespace Tracer.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    public struct CUDAVertex
    {
        public float3 Position;
    }
}
