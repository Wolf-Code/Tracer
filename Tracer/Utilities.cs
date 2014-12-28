using System.Drawing;
using System.IO;
using Tracer.Classes;
using Tracer.Classes.Util;

namespace Tracer
{
    public static class Utilities
    {
        public static Bitmap ConvertByteArray( byte [ ] Data )
        {
            using ( MemoryStream M = new MemoryStream( Data ) )
            {
                return ( Bitmap ) Image.FromStream( M );
            }
        }

        public static Vector3 RandomDirectionInSameDirection( Vector3 Original )
        {
            Vector3 New = new Vector3( RNG.GetUnitFloat( ), RNG.GetUnitFloat( ), RNG.GetUnitFloat( ) ).Normalized( );
            if ( New.Dot( Original ) < 0 )
                New *= -1;

            return New;
        }

        public static Vector3 RandomCosineDirectionInSameDirection( Vector3 Original )
        {
            Vector3 First = RandomDirectionInSameDirection( Original );
            First = ( First + Original ).Normalized( );

            return First;
        }

        public static Classes.Util.Color VectorToColor( Vector3 V )
        {
            return new Classes.Util.Color( V.X, V.Y, V.Z );
        }

        public static Vector3 ColorToVector( Classes.Util.Color C )
        {
            return new Vector3( C.R, C.G, C.B );
        }

        public static Vector3 Reflect( Vector3 V, Vector3 Normal )
        {
            return V - 2 * V.Dot( Normal ) * Normal;
        }
    }
}
