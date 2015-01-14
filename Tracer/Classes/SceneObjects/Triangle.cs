using System;
using System.ComponentModel;
using ManagedCuda.VectorTypes;
using Tracer.Classes.Util;
using Tracer.Enums.CUDA;
using Tracer.Structs.CUDA;

namespace Tracer.Classes.SceneObjects
{
    /// <summary>
    /// A triangle object.
    /// </summary>
    [Serializable]
    public class Triangle : GraphicsObject
    {
        /// <summary>
        /// The triangle's 1st vertex.
        /// </summary>
        [Description( "The first of 3 vertices." )]
        [Category( "Properties" )]
        public Vector3 Vertex1 { set; get; }

        /// <summary>
        /// The triangle's 1st vertex.
        /// </summary>
        [Description( "The second of 3 vertices." )]
        [Category( "Properties" )]
        public Vector3 Vertex2 { set; get; }

        /// <summary>
        /// The triangle's 1st vertex.
        /// </summary>
        [Description( "The third of 3 vertices." )]
        [Category( "Properties" )]
        public Vector3 Vertex3 { set; get; }

        public Triangle( )
        {
            Name = "Triangle";
            Vertex1 = new Vector3( -10, 0, 0 );
            Vertex2 = new Vector3( 0, 10, 0 );
            Vertex3 = new Vector3( 10, 0, 0 );
        }

        public Triangle( Vector3 V1, Vector3 V2, Vector3 V3 ) : this( )
        {
            Vertex1 = V1;
            Vertex2 = V2;
            Vertex3 = V3;
        }

        public override CUDAObject [ ] ToCUDA( )
        {
            Vector3 P1 = Vertex2 - Vertex1;
            Vector3 P2 = Vertex3 - Vertex1;
            float3 Cross = P2.Cross( P1 ).Normalized( ).ToFloat3( );

            return new [ ]
            {
                new CUDAObject
                {
                    Material = Material.ToCUDAMaterial( ),
                    Triangle = new CUDATriangleObject
                    {
                        V1 = new CUDAVertex
                        {
                            Position = Vertex1.ToFloat3( ),
                            Normal = Cross
                        },
                        V2 = new CUDAVertex
                        {
                            Position = Vertex2.ToFloat3( ),
                            Normal = Cross
                        },
                        V3 = new CUDAVertex
                        {
                            Position = Vertex3.ToFloat3( ),
                            Normal = Cross
                        },
                        Normal = Cross
                    },
                    Type = CUDAObjectType.Triangle
                }
            };
        }
    }
}