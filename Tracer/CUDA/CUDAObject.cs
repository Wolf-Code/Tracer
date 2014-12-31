using System.Runtime.InteropServices;

namespace Tracer.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    struct CUDAObject
    {
        public CUDAObjectType Type;
        public CUDASphereObject Sphere;
        public CUDAPlane Plane;
        public CUDAMaterial Material;
    }
}
