using System.Runtime.InteropServices;
using ManagedCuda.VectorTypes;

namespace Tracer.Structs.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    public struct CUDATriangleObject
    {
        public CUDAVertex V1, V2, V3;
        public float3 Normal;
    }
}