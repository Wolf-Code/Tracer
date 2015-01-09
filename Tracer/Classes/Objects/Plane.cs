using System;
using System.ComponentModel;
using Tracer.Classes.Util;
using Tracer.CUDA;

namespace Tracer.Classes.Objects
{
    /// <summary>
    /// A plane object.
    /// </summary>
    [Serializable]
    public class Plane : GraphicsObject
    {
        /// <summary>
        /// The plane's normal.
        /// </summary>
        [Category( "Properties" )]
        public Vector3 Normal { set; get; }

        /// <summary>
        /// The plane's distance from the center, along its normal.
        /// </summary>
        [Description( "The offset from the center, in its normal's direction." )]
        [Category( "Properties" )]
        public float Offset { set; get; }

        public Plane( )
        {
            this.Normal = new Vector3( 0, 1, 0 );
            this.Offset = 0;
            this.Name = "Plane";
        }

        public Plane( Vector3 Normal, float Offset )
        {
            this.Normal = Normal;
            this.Offset = Offset;
            this.Name = "Plane";
        }

        public override CUDAObject ToCUDA( )
        {
            return new CUDAObject
            {
                Material = this.Material.ToCUDAMaterial( ),
                Plane = new CUDAPlaneObject
                {
                    Normal = this.Normal.ToFloat3( ),
                    Offset = this.Offset
                },
                Type = CUDAObjectType.Plane
            };
        }
    }
}
