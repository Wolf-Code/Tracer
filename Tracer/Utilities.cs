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

        public static Classes.Util.Color VectorToColor( Vector3 V )
        {
            return new Classes.Util.Color( V.X, V.Y, V.Z );
        }

        public static Vector3 ColorToVector( Classes.Util.Color C )
        {
            return new Vector3( C.R, C.G, C.B );
        }
    }
}
