using System.Runtime.InteropServices;
using Tracer.Enums.CUDA;

namespace Tracer.Structs.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    public struct CUDAObject
    {
        public uint ID;
        public CUDAObjectType Type;
        public CUDASphereObject Sphere;
        public CUDAPlaneObject Plane;
        public CUDATriangleObject Triangle;
        public CUDAMeshObject Mesh;
        public CUDAMaterial Material;
    }
}