
using System.Collections.Generic;
using System.IO;
using System.Windows.Forms;
using Tracer.Classes;
using Tracer.Classes.Util;
using Tracer.Importers;

namespace Tracer.Models
{
    public class OBJMaterialData
    {
        class OBJMaterial
        {
            public float Ns;
            public float Ni;
            public float illum;
            public Color Ka = new Color( );
            public Color Kd = new Color( 1, 1, 1 );
            public Color Ks = new Color( );
            public Color Ke = new Color( );
        }

        private readonly Dictionary<string, OBJMaterial> Materials = new Dictionary<string, OBJMaterial>( );
        private string CurrentMaterial;

        public OBJMaterialData( string Path )
        {
            if ( !File.Exists( Path ) )
                return;

            using ( StreamReader Reader = new StreamReader( Path ) )
            {
                while ( !Reader.EndOfStream )
                    ParseLine( Reader.ReadLine( ) );
            }
        }

        public Material GetMaterial( string Name )
        {
            if ( !Materials.ContainsKey( Name ) )
                return new Material( );

            return new Material
            {
                Color = Materials[ Name ].Kd,
                Radiance = Materials[ Name ].Ke
            };
        }

        private void ParseLine( string Line )
        {
            Line = Line.Replace( '\t', ' ' );
            while ( Line.Contains( "  " ) )
                Line = Line.Replace( "  ", " " );

            string [ ] Data = Line.Trim( ).Split( );

            switch ( Data[ 0 ] )
            {
                case "newmtl":
                    CurrentMaterial = Data[ 1 ];
                    Materials.Add( CurrentMaterial, new OBJMaterial( ) );
                    break;

                case "Kd":
                    Materials[ CurrentMaterial ].Kd = ParseColor( Data );
                    break;
            
                case "Ke":
                    Materials[ CurrentMaterial ].Ke = ParseColor( Data );
                    break;
            }
        }

        private Color ParseColor( string [ ] Data )
        {
            float R = OBJImporter.ParseFloat( Data[ 1 ] );
            float G = OBJImporter.ParseFloat( Data[ 2 ] );
            float B = OBJImporter.ParseFloat( Data[ 3 ] );

            return new Color( R, G, B );
        }
    }
}
