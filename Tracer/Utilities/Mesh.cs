using System;
using Tracer.Classes.ModelData;
using Tracer.Classes.Util;

namespace Tracer.Utilities
{
    public static class Mesh
    {
        public static Tuple<Vector3, Vector3> AABB( ModelMesh M )
        {
            Vector3 Min = new Vector3( );
            Vector3 Max = new Vector3( );
            foreach ( Vertex V in M.Vertices )
            {
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

            return new Tuple<Vector3, Vector3>( Min, Max );
        }
    }
}