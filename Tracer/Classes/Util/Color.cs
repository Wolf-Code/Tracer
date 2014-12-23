
using System;

namespace Tracer.Classes.Util
{
    public struct Color
    {
        public static Color White
        {
            get { return new Color( 1f, 1f, 1f ); }
        }

        public static Color Black
        {
            get { return new Color( 0f, 0f, 0f ); }
        }

        public byte R;
        public byte G;
        public byte B;

        public float fR
        {
            get { return R / 255f; }
        }

        public float fG
        {
            get
            {
                return G / 255f;
            }
        }

        public float fB
        {
            get
            {
                return B / 255f;
            }
        }

        public System.Drawing.Color DrawingColor
        {
            get { return System.Drawing.Color.FromArgb( R, G, B ); }
        }

        public Color( byte R, byte G, byte B )
        {
            this.R = Math.Min( ( byte ) 255, R );
            this.G = Math.Min( ( byte ) 255, G );
            this.B = Math.Min( ( byte ) 255, B );
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

        #region Operators

        public static Color operator *( Color C1, float M )
        {
            return new Color( C1.fR * M, C1.fG * M, C1.fB * M );
        }

        public static Color operator /( Color C1, float M )
        {
            return new Color( C1.fR / M, C1.fG / M, C1.fB / M );
        }

        public static Color operator +( Color C1, Color C2 )
        {
            return new Color( C1.fR + C2.fR, C1.fG + C2.fG, C1.fB + C2.fB );
        }

        public static Color operator *( Color C1, Color C2 )
        {
            return new Color( C1.fR * C2.fR, C1.fG * C2.fG, C1.fB * C2.fB );
        }

        public static bool operator ==( Color C1, Color C2 )
        {
            return C1.R == C2.R && C1.G == C2.G && C1.B == C2.B;
        }

        public static bool operator !=( Color C1, Color C2 )
        {
            return !( C1 == C2 );
        }

        #endregion
    }
}
