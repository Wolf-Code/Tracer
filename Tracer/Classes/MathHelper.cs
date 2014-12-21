using System;

namespace Tracer.Classes
{
    public static class MathHelper
    {
        /// <summary>
        /// Converts an angle in degrees into radians.
        /// </summary>
        /// <param name="Degrees">The angle in degrees.</param>
        /// <returns>The angle in radians.</returns>
        public static float ToRadians( float Degrees )
        {
            // R = D * ( Pi / 180 )
            return Degrees * ( float ) ( Math.PI / 180f );
        }
    }
}
