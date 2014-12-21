using System.Drawing;
using System.IO;

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
    }
}
