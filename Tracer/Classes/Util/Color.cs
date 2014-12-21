
namespace Tracer.Classes.Util
{
    public struct Color
    {
        public byte R;
        public byte G;
        public byte B;

        public System.Drawing.Color DrawingColor
        {
            get { return System.Drawing.Color.FromArgb( R, G, B ); }
        }

        public Color( byte R, byte G, byte B )
        {
            this.R = R;
            this.G = G;
            this.B = B;
        }

        public Color( float R, float G, float B )
        {
            this.R = ( byte ) ( R * 255 );
            this.G = ( byte ) ( G * 255 );
            this.B = ( byte ) ( B * 255 );
        }

        public override string ToString( )
        {
            return string.Format( "R: {0}, G: {1}, B: {2}", this.R, this.G, this.B );
        }
    }
}
