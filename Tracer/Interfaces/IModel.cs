using Tracer.Classes;
using Tracer.Classes.Objects;

namespace Tracer.Interfaces
{
    public interface IModel
    {
        Vertex [ ] Vertices { get; }

        uint [ ] Indices { get; }

        Triangle [ ] ToTriangles( );
    }
}
