using System;
using System.ComponentModel;
using Tracer.Classes.Util;
using Tracer.Enums.CUDA;
using Tracer.Structs.CUDA;

namespace Tracer.Classes.SceneObjects
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
            Normal = new Vector3( 0, 1, 0 );
            Offset = 0;
            Name = "Plane";
        }

        public Plane( Vector3 Normal, float Offset )
        {
            this.Normal = Normal;
            this.Offset = Offset;
            Name = "Plane";
        }

        public override CUDAObject [ ] ToCUDA( )
        {
            return new [ ]
            {
                new CUDAObject
                {
                    Material = Material.ToCUDAMaterial( ),
                    Plane = new CUDAPlaneObject
                    {
                        Normal = Normal.ToFloat3( ),
                        Offset = Offset
                    },
                    Type = CUDAObjectType.Plane
                }
            };
        }
    }
}