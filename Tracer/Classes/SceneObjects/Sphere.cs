using System;
using System.ComponentModel;
using Tracer.Classes.Util;
using Tracer.Enums.CUDA;
using Tracer.Structs.CUDA;

namespace Tracer.Classes.SceneObjects
{
    /// <summary>
    /// A sphere object.
    /// </summary>
    [Serializable]
    public class Sphere : GraphicsObject
    {
        /// <summary>
        /// The sphere's position.
        /// </summary>
        [Category( "Properties" )]
        public Vector3 Center { set; get; }

        /// <summary>
        /// The sphere's radius.
        /// </summary>
        [Category( "Properties" )]
        public float Radius { set; get; }

        public Sphere( )
        {
            this.Center = new Vector3( 0, 0, 0 );
            this.Radius = 20f;
            this.Name = "Sphere";
        }

        public Sphere( Vector3 Position, float Radius )
        {
            this.Center = Position;
            this.Radius = Radius;
            this.Name = "Sphere";
        }

        public override CUDAObject [ ] ToCUDA( )
        {
            return new [ ]
            {
                new CUDAObject
                {
                    Material = this.Material.ToCUDAMaterial( ),
                    Sphere = new CUDASphereObject
                    {
                        Position = this.Center.ToFloat3( ),
                        Radius = this.Radius
                    },
                    Type = CUDAObjectType.Sphere
                }
            };
        }
    }
}