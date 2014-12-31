using System.Collections.Generic;
using System.Linq;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;

namespace Tracer.Classes
{
    public static class RayCaster
    {
        public static Color BackgroundColor = new Color( 0, 0, 0.2f );
        public static List<GraphicsObject> Objects = new List<GraphicsObject>( );
        public static List<Light> Lights = new List<Light>( );
        public static uint MaxDepth = 2;
        public static uint Samples = 250;

        /// <summary>
        /// Casts the given ray and returns the color belonging to this ray.
        /// </summary>
        /// <param name="R">The ray to cast.</param>
        /// <returns>The color belonging to it.</returns>
        public static CollisionResult Trace( Ray R )
        {
            return
                // Get all collision results
                Objects.Select( O => O.CheckCollision( R ) )
                // Only take the ones which hit
                    .Where( O => O.Hit )
                // Order them by their distance
                    .OrderBy( O => O.Distance )
                // And grab the first ( or default if there is none )
                    .FirstOrDefault( );
        }

        /// <summary>
        /// Casts the given ray and returns the color belonging to this ray.
        /// </summary>
        /// <param name="R">The ray to cast.</param>
        /// <returns>The color belonging to it.</returns>
        public static Color Cast( Ray R )
        {
            return Effects.Radiance( R );
            //Effects.Calculate( R, Trace( R ) ); //!Res.Hit ? BackgroundColor : Effects.Calculate( R, Res );
        }

        public static Color Cast( int X, int Y )
        {
            Ray R = Renderer.Cam.GetRay( X, Y );
            Color C = Color.Black;
            for ( int Q = 0; Q < Samples; Q++ )
                C += Effects.Radiance2( R );

            return C / Samples;
        }
    }
}
