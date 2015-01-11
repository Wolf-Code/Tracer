using System.Collections.Generic;
using Tracer.Classes;
using Tracer.Classes.Objects;
using Tracer.Interfaces;

namespace Tracer.Models
{
    public class OBJModel : IModel
    {
        public Vertex [ ] Vertices { private set; get; }
        public uint [ ] Indices { private set; get; }

        public OBJModel( Vertex [ ] Vertices, uint [ ] Indices )
        {
            this.Vertices = Vertices;
            this.Indices = Indices;
        }

        public Triangle [ ] ToTriangles( )
        {
            List<Triangle> Ts = new List<Triangle>( );
            for ( int Q = 0; Q < Indices.Length; Q += 3 )
            {
                Vertex V1 = Vertices[ Indices[ Q ] ];
                Vertex V2 = Vertices[ Indices[ Q + 1 ] ];
                Vertex V3 = Vertices[ Indices[ Q + 2 ] ];

                Triangle T = new Triangle( V1.Position, V2.Position, V3.Position );
                Ts.Add( T );
            }

            return Ts.ToArray( );
        }
    }
}
