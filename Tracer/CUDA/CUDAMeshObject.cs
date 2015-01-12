﻿using System.Runtime.InteropServices;
using ManagedCuda.BasicTypes;

namespace Tracer.CUDA
{
    [StructLayout( LayoutKind.Sequential )]
    public struct CUDAMeshObject
    {
        public CUdeviceptr TrianglesPointer;
        public CUDASphereObject BoundingVolume;
        public uint TriangleCount;
    }
}
