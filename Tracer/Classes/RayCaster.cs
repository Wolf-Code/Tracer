using System.Collections.Generic;
using System.Linq;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;

namespace Tracer.Classes
{
    public static class RayCaster
    {
        public static Color BackgroundColor = new Color( 0, 0, 0 );
        public static List<GraphicsObject> Objects = new List<GraphicsObject>( ); 

        /// <summary>
        /// Casts the given ray and returns the color belonging to this ray.
        /// </summary>
        /// <param name="R">The ray to cast.</param>
        /// <returns>The color belonging to it.</returns>
        public static Color Cast( Ray R )
        {
            CollisionResult Res =
                // Get all collision results
                Objects.Select( O => O.CheckCollision( R ) )
                    // Only take the ones which hit
                    .Where( O => O.Hit )
                    // Order them by their distance
                    .OrderBy( O => O.Distance )
                    // And grab the first ( or default if there is none )
                    .FirstOrDefault( );

            return Res.Hit ? Res.Object.Material.Color : BackgroundColor;
        }
    }
}
