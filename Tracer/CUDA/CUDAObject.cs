using System.Runtime.InteropServices;

namespace Tracer.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    public struct CUDAObject
    {
        public uint ID;
        public CUDAObjectType Type;
        public CUDASphereObject Sphere;
        public CUDAPlane Plane;
        public CUDAMaterial Material;
    }
}
