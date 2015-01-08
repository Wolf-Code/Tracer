﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Design;
using System.Linq;
using Tracer.Classes.Util;
using Tracer.CUDA;

using SceneCUDAData = System.Tuple<Tracer.CUDA.CUDAObject[],Tracer.CUDA.CUDAObject[]>;

namespace Tracer.Classes.Objects
{
    [Serializable]
    public class Scene
    {
        [Category( "Rendering" )]
        public Camera Camera { set; get; }

        [Editor( typeof ( ControlCollectionEditor ), typeof ( UITypeEditor ) )]
        [Category( "Geometry" )]
        [Description( "The objects in the scene." )]
        public List<GraphicsObject> Objects { set; get; }

        #region Default

        public static Scene Default
        {
            get
            {
                Scene Scn = new Scene
                {
                    Camera =
                    {
                        Position = new Vector3( 0, 90, 80 ),
                        Angle = new Angle { Pitch = 0, Yaw = 0, Roll = 0f }
                    }
                };


                Sphere Light = Scn.AddSphere( new Vector3( 0, 60, 0 ), 5 );
                Light.Name = "Light";
                Light.Material.Radiance = new Util.Color( 4000, 4000, 4000 );

                Plane Floor = Scn.AddPlane( new Vector3( 0, 1, 0 ), 0 );
                Floor.Name = "Floor";
                Floor.Material.Color = new Util.Color( 1f, 1f, 1f );

                Plane Front = Scn.AddPlane( new Vector3( 0, 0, 1 ), 90 );
                Front.Name = "Front";
                Front.Material.Color = new Util.Color( 1f, 1f, 1f );

                Plane Back = Scn.AddPlane( new Vector3( 0, 0, -1 ), 90 );
                Back.Name = "Back";
                Back.Material.Color = new Util.Color( 1f, 1f, 1f );

                Plane Ceiling = Scn.AddPlane( new Vector3( 0, -1, 0 ), 180 );
                Ceiling.Name = "Ceiling";
                Ceiling.Material.Color = new Util.Color( 1f, 1f, 1f );

                Plane Left = Scn.AddPlane( new Vector3( 1, 0, 0 ), 90 );
                Left.Name = "Left";
                Left.Material.Color = new Util.Color( 1f, .5f, .5f );

                Plane Right = Scn.AddPlane( new Vector3( -1, 0, 0 ), 90 );
                Right.Name = "Right";
                Right.Material.Color = new Util.Color( .5f, .5f, 1f );

                Sphere WhiteSphere = Scn.AddSphere( new Vector3( -40, 30, -40 ), 30 );
                WhiteSphere.Name = "White sphere";
                WhiteSphere.Material.Color = new Util.Color( 1f, 1f, 1f );

                Sphere MirrorSphere = Scn.AddSphere( new Vector3( 40, 20, -20 ), 20 );
                MirrorSphere.Name = "Mirror sphere";
                MirrorSphere.Material.Type = CUDAMaterialType.Reflective;
                MirrorSphere.Material.Glossyness = 0f;

                return Scn;
            }
        }

        #endregion

        public Scene( )
        {
            this.Objects = new List<GraphicsObject>( );
            this.Camera = new Camera( );
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

        private CUDAObject ConvertToCUDA( GraphicsObject G )
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

            return O;
        }

        public SceneCUDAData ToCUDA( )
        {
            List<CUDAObject> Obj = new List<CUDAObject>( );
            List<CUDAObject> Lights = new List<CUDAObject>( );
            foreach ( GraphicsObject G in Objects.Where( O => O.Enabled ) )
            {
                CUDAObject O = this.ConvertToCUDA( G );
                O.ID = ( uint ) Objects.IndexOf( G );
                Obj.Add( O );

                if ( G.Material.Radiance.R > 0 || G.Material.Radiance.G > 0 || G.Material.Radiance.B > 0 )
                {
                    CUDAObject O2 = this.ConvertToCUDA( G );
                    O2.ID = ( uint ) Objects.IndexOf( G );
                    Lights.Add( O2 );
                }
            }

            return new SceneCUDAData( Obj.ToArray(  ), Lights.ToArray(  ) );
        }
    }
}