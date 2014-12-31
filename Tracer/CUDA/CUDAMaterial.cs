using System.Runtime.InteropServices;
using ManagedCuda.VectorTypes;

namespace Tracer.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    public struct CUDAMaterial
    {
        public float3 Color;
        public float3 Radiance;
        public CUDAMaterialType Type;
        public float Glossyness;
    }
}
