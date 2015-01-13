using System;
using System.Collections.Generic;
using ManagedCuda;
using Tracer.Classes.SceneObjects;
using Tracer.Classes.Util;
using Tracer.Enums.CUDA;
using Tracer.Interfaces;
using Tracer.Structs.CUDA;

namespace Tracer.Classes.ModelData
{
    public class ModelMesh
    {
        public Vertex [ ] Vertices { private set; get; }
        public Material Material { private set; get; }
        public string Name;
        private IModel Parent;

        public ModelMesh( Vertex [ ] Vertices )
        {
            this.Vertices = Vertices;
            this.Material = new Material( );
        }

        public void SetParent( IModel Model )
        {
            this.Parent = Model;
        }

        public CUDAObject ToCUDA( )
        {
            List<CUDATriangleObject> Ts = new List<CUDATriangleObject>( );
            for ( int Q = 0; Q < Vertices.Length; Q += 3 )
            {
                Vertex V1 = Vertices[ Q ];
                Vertex V2 = Vertices[ Q + 1 ];
                Vertex V3 = Vertices[ Q + 2 ];

                Triangle T = new Triangle(
                    Parent.Position + V1.Position,
                    Parent.Position + V2.Position,
                    Parent.Position + V3.Position );

                Ts.Add( T.ToCUDA( )[ 0 ].Triangle );
            }

            CudaDeviceVariable<CUDATriangleObject> Triangles = new CudaDeviceVariable<CUDATriangleObject>( Ts.Count );
            Triangles.CopyToDevice( Ts.ToArray( ) );
            Tuple<Vector3, Vector3> MinMax = Utilities.Mesh.AABB( this );

            return new CUDAObject
            {
                Mesh = new CUDAMeshObject
                {
                    TriangleCount = ( uint ) Ts.Count,
                    TrianglesPointer = Triangles.DevicePointer,
                    BoundingVolume =
                        new Sphere( ( MinMax.Item1 + MinMax.Item2 ) / 2, ( MinMax.Item2 - MinMax.Item1 ).Length ).ToCUDA
                            ( )[
                                0 ].Sphere
                },
                Material = this.Material.ToCUDAMaterial( ),
                Type = CUDAObjectType.MeshType
            };
        }
    }
}