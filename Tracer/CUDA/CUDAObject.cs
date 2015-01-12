using System.Runtime.InteropServices;

namespace Tracer.CUDA
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
