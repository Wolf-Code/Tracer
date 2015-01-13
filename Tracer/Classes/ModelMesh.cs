
using Tracer.CUDA;

namespace Tracer.Classes
{
    public class ModelMesh
    {
        private Vertex [ ] Vertices;
        private Material Material;

        public CUDAMeshObject ToCUDA( )
        {
            return new CUDAMeshObject( );
        }
    }
}
