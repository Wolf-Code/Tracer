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
            Center = new Vector3( 0, 0, 0 );
            Radius = 20f;
            Name = "Sphere";
        }

        public Sphere( Vector3 Position, float Radius )
        {
            Center = Position;
            this.Radius = Radius;
            Name = "Sphere";
        }

        public override CUDAObject [ ] ToCUDA( )
        {
            return new [ ]
            {
                new CUDAObject
                {
                    Material = Material.ToCUDAMaterial( ),
                    Sphere = new CUDASphereObject
                    {
                        Position = Center.ToFloat3( ),
                        Radius = Radius
                    },
                    Type = CUDAObjectType.Sphere
                }
            };
        }
    }
}