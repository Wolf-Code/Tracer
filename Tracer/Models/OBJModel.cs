﻿using System;
using System.Collections.Generic;
using System.Linq;
using Tracer.Classes;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;
using Tracer.CUDA;
using Tracer.Interfaces;

namespace Tracer.Models
{
    public class OBJModel : IModel
    {
        public ModelMesh [ ] Meshes { private set; get; }
        public Vector3 Position { private set; get; }
        public Vector3 Scale { private set; get; }
        private Vector3 Min, Max;

        public OBJModel( ModelMesh [ ] Meshes )
        {
            this.Meshes = Meshes;
            foreach ( ModelMesh M in this.Meshes )
                M.SetParent( this );

            Tuple<Vector3, Vector3> MinMax = Utilities.Mesh.AABB( Meshes[ 0 ] );
            Min = MinMax.Item1;
            Max = MinMax.Item2;

            for ( int Q = 1; Q < Meshes.Length; Q++ )
            {
                MinMax = Utilities.Mesh.AABB( Meshes[ Q ] );
                Vector3 TempMin = MinMax.Item1;
                Vector3 TempMax = MinMax.Item2;

                Min.X = Math.Min( TempMin.X, Min.X );
                Min.Y = Math.Min( TempMin.Y, Min.Y );
                Min.Z = Math.Min( TempMin.Z, Min.Z );

                Max.X = Math.Min( TempMax.X, Max.X );
                Max.Y = Math.Min( TempMax.Y, Max.Y );
                Max.Z = Math.Min( TempMax.Z, Max.Z );
            }
        }

        public Sphere BoundingSphere( )
        {
            return new Sphere( ( Min + Max ) / 2, ( Max - Min ).Length );
        }

        public CUDAObject [ ] ToCuda( )
        {
            return this.Meshes.Select( O => O.ToCUDA( ) ).ToArray( );
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
