using System.Collections.Generic;
using Tracer.Classes;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;
using Tracer.Interfaces;

namespace Tracer.Models
{
    public class OBJModel : IModel
    {
        public Vertex [ ] Vertices { private set; get; }
        private readonly Vector3 Center;
        private Vector3 Position;
        private Vector3 Scale;
        private Vector3 Min, Max;

        public OBJModel( Vertex [ ] Vertices )
        {
            Center = new Vector3( );

            this.Vertices = Vertices;
            Min = new Vector3( );
            Max = new Vector3( );
            foreach ( Vertex V in Vertices )
            {
                Center += V.Position;
                if ( V.Position.X < Min.X )
                    Min.X = V.Position.X;

                if ( V.Position.Y < Min.Y )
                    Min.Y = V.Position.Y;

                if ( V.Position.Z < Min.Z )
                    Min.Z = V.Position.Z;



                if ( V.Position.X > Max.X )
                    Max.X = V.Position.X;

                if ( V.Position.Y > Max.Y )
                    Max.Y = V.Position.Y;

                if ( V.Position.Z > Max.Z )
                    Max.Z = V.Position.Z;
            }

            Center /= Vertices.Length;
        }

        public Sphere BoundingSphere( )
        {
            return new Sphere( Center, ( Max - Min ).Length );
        }

        public Triangle [ ] ToTriangles( )
        {
            List<Triangle> Ts = new List<Triangle>( );
            for ( int Q = 0; Q < Vertices.Length; Q += 3 )
            {
                Vertex V1 = Vertices[ Q ];
                Vertex V2 = Vertices[ Q + 1 ];
                Vertex V3 = Vertices[ Q + 2 ];

                Triangle T = new Triangle( this.Position + V1.Position, this.Position + V2.Position, this.Position + V3.Position );
                Ts.Add( T );
            }

            return Ts.ToArray( );
        }

        public void SetPosition( Vector3 Position )
        {
            this.Position = Position;
        }

        public void SetScale( Vector3 Scale )
        {
            this.Scale = Scale;
        }
    }
}
