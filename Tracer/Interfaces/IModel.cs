using Tracer.Classes.ModelData;
using Tracer.Classes.SceneObjects;
using Tracer.Classes.Util;
using Tracer.Structs.CUDA;

namespace Tracer.Interfaces
{
    public interface IModel
    {
        ModelMesh [ ] Meshes { get; }
        Vector3 Position { get; }
        Vector3 Scale { get; }

        CUDAObject [ ] ToCuda( );
        void SetPosition( Vector3 Position );
        void SetScale( Vector3 Scale );

        Sphere BoundingSphere( );
    }
}