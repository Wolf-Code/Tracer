using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Tracer.Classes;
using Tracer.Classes.Util;
using Tracer.Interfaces;
using Tracer.Models;

namespace Tracer.Importers
{
    class OBJImporter : IModelImporter
    {
        private readonly List<Vertex> Vertices = new List<Vertex>( );
        private readonly List<uint> Indices = new List<uint>( );

        public IModel Import( string Path )
        {
            using ( StreamReader Reader = new StreamReader( Path ) )
            {
                while ( !Reader.EndOfStream )
                {
                    string Line = Reader.ReadLine( ).Replace( '\t', ' ' );
                    while ( Line.Contains( "  " ) ) Line = Line.Replace( "  ", " " );

                    string [ ] Data = Line.Split( new [ ] { " " }, StringSplitOptions.RemoveEmptyEntries );

                    switch ( Data[ 0 ] )
                    {
                        case "v":
                            AddVertex( Data );
                            break;

                        case "f":
                            ParseFaces( Data );
                            break;
                    }
                }
            }

            OBJModel Model = new OBJModel( Vertices.ToArray( ), Indices.ToArray( ) );
            Vertices.Clear( );
            Indices.Clear( );
            return Model;
        }

        private uint GetProperID( int ParsedID )
        {
            if ( ParsedID < 0 )
                ParsedID = Vertices.Count + ParsedID;

            return ( uint ) ParsedID;
        }

        private void ParseFaces( string [ ] Data )
        {
            uint BaseID = GetProperID( int.Parse( Data[ 1 ] ) );

            for ( int Q = 2; Q <= Data.Length - 2; Q++ )
            {
                Indices.Add( BaseID );
                for ( int I = 0; I < 2; I++ )
                {
                    int ID = int.Parse( Data[ Q + I ] );

                    Indices.Add( GetProperID( ID ) );
                }
            }
        }

        private void AddVertex( string [ ] Data )
        {
            float X = float.Parse( Data[ 1 ], CultureInfo.InvariantCulture );
            float Y = float.Parse( Data[ 2 ], CultureInfo.InvariantCulture );
            float Z = float.Parse( Data[ 3 ], CultureInfo.InvariantCulture );

            Vertices.Add( new Vertex
            {
                Position = new Vector3( X, Y, Z )
            } );
        }
    }
}
