using System;
using System.Collections.Generic;
using System.ComponentModel;
using Tracer.Classes.Util;
using Tracer.CUDA;

namespace Tracer.Classes.Objects
{
    [Serializable]
    public class Scene
    {
        [Editor( typeof( ControlCollectionEditor ),
            typeof( System.Drawing.Design.UITypeEditor ) )]
        public List<GraphicsObject> Objects { set; get; }

        public Scene( )
        {
            this.Objects = new List<GraphicsObject>( );
        }

        public Sphere AddSphere( Vector3 Position, float Radius )
        {
            Sphere S = new Sphere( Position, Radius );
            Objects.Add( S );

            return S;
        }

        public Plane AddPlane( Vector3 Normal, float Offset )
        {
            Plane P = new Plane( Normal, Offset );
            Objects.Add( P );

            return P;
        }

        public CUDAObject [ ] ToCUDA( )
        {
            List<CUDAObject> Obj = new List<CUDAObject>( );
            foreach ( GraphicsObject G in Objects )
            {
                CUDAObject O = new CUDAObject { Material = G.Material.ToCUDAMaterial( ) };

                if ( G is Sphere )
                {
                    O.Sphere = ( G as Sphere ).ToCUDASphere( );
                    O.Type = CUDAObjectType.Sphere;
                }

                if ( G is Plane )
                {
                    O.Plane = ( G as Plane ).ToCUDAPlane( );
                    O.Type = CUDAObjectType.Plane;
                }

                Obj.Add( O );
            }

            return Obj.ToArray( );
        }
    }
}
