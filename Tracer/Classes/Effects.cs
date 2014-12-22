
using System;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;

namespace Tracer.Classes
{
    public static class Effects
    {
        /// <summary>
        /// The bias for shadow checking.
        /// </summary>
        public const float ShadowBias = 1f;

        /// <summary>
        /// Calculates the color of the position the <see cref="Ray"/> hit.
        /// </summary>
        /// <param name="R">The ray.</param>
        /// <param name="Res">The <see cref="CollisionResult"/>.</param>
        /// <returns>The <see cref="Color"/>.</returns>
        public static Color MaterialColor( Ray R, CollisionResult Res )
        {
            return Res.Object.Material.Color;
        }

        /// <summary>
        /// Calculates whether a given collision result is in a given light's shadow.
        /// </summary>
        /// <param name="Res"></param>
        /// <param name="L"></param>
        /// <returns></returns>
        public static bool IsInShadow( CollisionResult Res, Light L )
        {
            Ray R = new Ray
            {
                Start = Res.Position + Res.Normal * ShadowBias,
                Direction = ( L.Position - Res.Position ).Normalized( )
            };

            CollisionResult ShadowRes = RayCaster.Trace( R );

            return ShadowRes.Hit;
        }

        public static float ShadowMultiplier( bool InShadow )
        {
            return ( InShadow ? 0.1f : 1f );
        }

        public static Color DiffuseLightColor( Ray R, CollisionResult Res )
        {
            Color C = new Color( 0, 0, 0 );
            foreach ( Light L in RayCaster.Lights )
            {
                bool Shadowed = IsInShadow( Res, L );

                Vector3 LightDirection = L.Position - Res.Position;
                float Length = LightDirection.Length;
                float DistDiv = ( L.FallOffDistance - Length ) / L.FallOffDistance;
                if ( DistDiv < 0 )
                    DistDiv = 0;

                if ( DistDiv > 1f )
                    DistDiv = 1f;

                LightDirection /= Length;
                float Intensity = DistDiv * Math.Max( 0, Res.Normal.Dot( LightDirection ) ) * L.Intensity;
                float ShadowMul = ShadowMultiplier( Shadowed );
                C += L.DiffuseColor * Intensity * ShadowMul;
            }

            return C;
        }
    }
}
