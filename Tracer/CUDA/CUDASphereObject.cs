using System.Runtime.InteropServices;
using ManagedCuda.VectorTypes;

namespace Tracer.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    struct CUDASphereObject
    {
        public float3 Position;
        public float Radius;
    }
}
