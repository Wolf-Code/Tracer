using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Tracer.Classes.ModelData;
using Tracer.Classes.Util;
using Tracer.Interfaces;
using Tracer.Models;
using Tracer.Structs;

namespace Tracer.Importers
{
    internal class OBJImporter : IModelImporter
    {
        private readonly List<Vector3> Positions = new List<Vector3>( );
        private readonly List<Vector3> Normals = new List<Vector3>( );
        private readonly List<Vector2> TextureCoordinates = new List<Vector2>( );
        private readonly List<FaceData> Faces = new List<FaceData>( );
        private readonly List<ModelMesh> Meshes = new List<ModelMesh>( );
        private string CurrentMesh = string.Empty;

        public IModel Import( string Path )
        {
            using ( StreamReader Reader = new StreamReader( Path ) )
            {
                while ( !Reader.EndOfStream )
                {
                    string Line = Reader.ReadLine( ).Replace( '\t', ' ' );
                    while ( Line.Contains( "  " ) ) Line = Line.Replace( "  ", " " );

                    string [ ] Data = Line.Split( new [ ] { " " }, StringSplitOptions.RemoveEmptyEntries );
                    if ( Data.Length <= 0 )
                        continue;

                    switch ( Data[ 0 ] )
                    {
                        case "v":
                            AddVertex( Data );
                            break;

                        case "f":
                            ParseFaceLine( Data );
                            break;

                        case "g":
                            if ( Faces.Count > 0 )
                            {
                                ModelMesh M = CreateModelMesh( );
                                M.Name = CurrentMesh;

                                Meshes.Add( M );
                            }

                            CurrentMesh = string.Join( " ", Data );
                            break;
                    }
                }
            }

            if ( Faces.Count > 0 )
            {
                ModelMesh M = CreateModelMesh( );
                M.Name = CurrentMesh;
                Meshes.Add( M );
            }


            OBJModel Model = new OBJModel( Meshes.ToArray( ) );

            return Model;
        }

        private ModelMesh CreateModelMesh( )
        {
            List<Vertex> Vertices = new List<Vertex>( );
            for ( int Q = 0; Q < Faces.Count; Q++ )
            {
                Vector3 Pos = Positions[ Faces[ Q ].Position ];
                Vector3 Norm = Normals.Count > 0 ? Normals[ Faces[ Q ].Normal ] : new Vector3( 0, 0, 0 );
                Vector2 UV = TextureCoordinates.Count > 0
                    ? TextureCoordinates[ Faces[ Q ].TextureCoordinate ]
                    : new Vector2( 0, 0 );

                Vertex V = new Vertex
                {
                    Position = Pos,
                    Normal = Norm,
                    UV = UV
                };

                Vertices.Add( V );
            }

            Faces.Clear( );

            return new ModelMesh( Vertices.ToArray( ) );
        }

        private float ParseFloat( string Data )
        {
            return float.Parse( Data, CultureInfo.InvariantCulture );
        }

        private int ParseInt( string Data )
        {
            return int.Parse( Data );
        }

        private int GetProperID( int ParsedID, ICollection Collection )
        {
            // If the parsed ID is negative, it means we take vertex n - id
            if ( ParsedID < 0 )
                ParsedID = Collection.Count + ParsedID + 1;

            return ParsedID - 1;
        }

        private FaceData ParseFaceData( string Data )
        {
            int SlashCount = Data.Count( O => O == '/' );

            if ( SlashCount == 0 )
                return ParseFaceIndices( Data );

            if ( SlashCount == 1 )
                return ParseFaceIndices_TextureCoordinates( Data );

            bool DoubleSlash = Data.IndexOf( "//" ) >= 0;

            return DoubleSlash ? ParseFaceIndices_Normals( Data ) : ParseFaceIndices_TextureCoordinates_Normals( Data );
        }

        private void ParseFaceLine( string [ ] Data )
        {
            FaceData First = ParseFaceData( Data[ 1 ] );
            for ( int Q = 2; Q <= Data.Length - 2; Q++ )
            {
                Faces.Add( First );
                for ( int I = 0; I < 2; I++ )
                {
                    FaceData D = ParseFaceData( Data[ Q + I ] );

                    Faces.Add( D );
                }
            }
        }

        private FaceData ParseFaceIndices_Normals( string Data )
        {
            // v1//n1 v2//n2 
            string [ ] SubData = Data.Split( new [ ] { "//" }, StringSplitOptions.RemoveEmptyEntries );
            int V = GetProperID( ParseInt( SubData[ 0 ] ), Positions );
            int N = GetProperID( ParseInt( SubData[ 1 ] ), Normals );

            return new FaceData
            {
                Position = V,
                Normal = N
            };
        }

        private FaceData ParseFaceIndices_TextureCoordinates_Normals( string Data )
        {
            // v1/t1/n1 v2/t2/n2
            string [ ] SubData = Data.Split( '/' );
            int V = GetProperID( ParseInt( SubData[ 0 ] ), Positions );
            int T = GetProperID( ParseInt( SubData[ 1 ] ), TextureCoordinates );
            int N = GetProperID( ParseInt( SubData[ 2 ] ), Normals );

            return new FaceData
            {
                Position = V,
                TextureCoordinate = T,
                Normal = N
            };
        }

        private FaceData ParseFaceIndices_TextureCoordinates( string Data )
        {
            // v1/t1 v2/t2
            string [ ] SubData = Data.Split( '/' );
            int V = GetProperID( ParseInt( SubData[ 0 ] ), Positions );
            int T = GetProperID( ParseInt( SubData[ 1 ] ), TextureCoordinates );

            return new FaceData
            {
                Position = V,
                TextureCoordinate = T
            };
        }

        private FaceData ParseFaceIndices( string Data )
        {
            // v1 v2 v3 v4 v5 
            int V = GetProperID( int.Parse( Data ), Positions );

            return new FaceData
            {
                Position = V
            };
        }

        private void AddVertex( string [ ] Data )
        {
            float X = ParseFloat( Data[ 1 ] );
            float Y = ParseFloat( Data[ 2 ] );
            float Z = ParseFloat( Data[ 3 ] );

            Positions.Add( new Vector3( X, Y, Z ) );
        }
    }
}