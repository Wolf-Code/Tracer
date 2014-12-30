using System.Runtime.InteropServices;

namespace Tracer.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    struct CUDAObject
    {
        public CUDATypeClass Type;
        public CUDASphereObject Sphere;
        public CUDAPlane Plane;
        public CUDAMaterial Material;
    }
}
