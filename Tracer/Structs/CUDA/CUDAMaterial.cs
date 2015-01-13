using System.Runtime.InteropServices;
using ManagedCuda.VectorTypes;
using Tracer.Enums.CUDA;

namespace Tracer.Structs.CUDA
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