
namespace Tracer.Classes.Util
{
    public struct Color
    {
        public float R;
        public float G;
        public float B;

        public System.Drawing.Color DrawingColor
        {
            get
            {
                return System.Drawing.Color.FromArgb(
                    ( int ) ( R * 255f ),
                    ( int ) ( G * 255f ),
                    ( int ) ( B * 255f ) );
            }
        }

        public Color( float R, float G, float B )
        {
            this.R = R;
            this.G = G;
            this.B = B;
        }

        public Color( byte R, byte G, byte B )
        {
            this.R = R / 255f;
            this.G = G / 255f;
            this.B = B / 255f;
        }

        public override string ToString( )
        {
            return string.Format( "R: {0}, G: {1}, B: {2}", this.R, this.G, this.B );
        }
    }
}
