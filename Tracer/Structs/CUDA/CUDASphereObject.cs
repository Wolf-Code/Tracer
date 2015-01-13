using System.Runtime.InteropServices;
using ManagedCuda.VectorTypes;

namespace Tracer.Structs.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    public struct CUDASphereObject
    {
        public float3 Position;
        public float Radius;
    }
}