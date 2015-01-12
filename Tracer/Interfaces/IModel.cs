using Tracer.Classes;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;

namespace Tracer.Interfaces
{
    public interface IModel
    {
        Vertex [ ] Vertices { get; }

        Triangle [ ] ToTriangles( );

        void SetPosition( Vector3 Position );
        void SetScale( Vector3 Scale );

        Sphere BoundingSphere( );
    }
}
