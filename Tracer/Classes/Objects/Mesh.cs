using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows.Forms.VisualStyles;
using ManagedCuda;
using Tracer.Classes.Util;
using Tracer.CUDA;
using Tracer.Importers;
using Tracer.Interfaces;

namespace Tracer.Classes.Objects
{
    [Serializable]
    public class Mesh : GraphicsObject
    {
        [Editor( typeof ( System.Windows.Forms.Design.FileNameEditor ), typeof ( System.Drawing.Design.UITypeEditor ) )]
        [Description( "The path to the model file" )]
        public string Path { set; get; }

        public Vector3 Position { set; get; }

        public Vector3 Scale { set; get; }

        public Mesh( )
        {
            this.Position = new Vector3( );
            this.Scale = new Vector3( 1, 1, 1 );
        }

        public override CUDAObject [ ] ToCUDA( )
        {
            IModel M = ModelImporter.Load( Path );
            M.SetPosition( this.Position );
            M.SetScale( this.Scale );

            Triangle [ ] Triangles = M.ToTriangles( );

            List<CUDAObject> Objs = new List<CUDAObject>( );
            foreach ( CUDAObject [ ] TObjs in Triangles.Select( T => T.ToCUDA( ) ) )
            {
                Objs.AddRange( TObjs );
            }

            CudaDeviceVariable<CUDATriangleObject> Ts = new CudaDeviceVariable<CUDATriangleObject>( Objs.Count );
            Ts.CopyToDevice( Objs.Select( O => O.Triangle ).ToArray( ) );

            CUDAMeshObject Mesh = new CUDAMeshObject
            {
                BoundingVolume = M.BoundingSphere( ).ToCUDA( )[ 0 ].Sphere,
                TriangleCount = ( uint ) Objs.Count,
                TrianglesPointer = Ts.DevicePointer
            };

            return new [ ]
            {
                new CUDAObject
                {
                    Material = this.Material.ToCUDAMaterial(  ),
                    Type = CUDAObjectType.MeshType,
                    Mesh = Mesh
                }
            };
        }
    }
}