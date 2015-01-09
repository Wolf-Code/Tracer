using System;
using System.ComponentModel;
using Tracer.Classes.Util;
using Tracer.CUDA;

namespace Tracer.Classes.Objects
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
            this.Name = "Triangle";
            this.Vertex1 = new Vector3( -10, 0, 0 );
            this.Vertex2 = new Vector3( 0, 10, 0 );
            this.Vertex3 = new Vector3( 10, 0, 0 );
        }

        public Triangle( Vector3 V1, Vector3 V2, Vector3 V3 ) : this( )
        {
            this.Vertex1 = V1;
            this.Vertex2 = V2;
            this.Vertex3 = V3;
        }

        public override CUDAObject ToCUDA( )
        {
            Vector3 P1 = this.Vertex2 - this.Vertex1;
            Vector3 P2 = this.Vertex3 - this.Vertex1;
            Vector3 Cross = P2.Cross( P1 );
            Console.WriteLine("P1 {0}, P2 {1}, Cross {2}", P1, P2, Cross.Normalized(  ));
            return new CUDAObject
            {
                Material = this.Material.ToCUDAMaterial( ),
                Triangle = new CUDATriangleObject
                {
                    V1 = new CUDAVertex
                    {
                        Position = this.Vertex1.ToFloat3( )
                    },
                    V2 = new CUDAVertex
                    {
                        Position = this.Vertex2.ToFloat3( )
                    },
                    V3 = new CUDAVertex
                    {
                        Position = this.Vertex3.ToFloat3( )
                    },
                    Normal = Cross.Normalized(  ).ToFloat3(  )
                },
                Type = CUDAObjectType.Triangle
            };
        }
    }
}
