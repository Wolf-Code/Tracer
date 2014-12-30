
using System;
using System.Drawing.Drawing2D;
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

            return ShadowRes.Distance < ( R.Start - L.Position ).Length;
        }

        public static float ShadowMultiplier( bool InShadow )
        {
            return ( InShadow ? 0.4f : 1f );
        }

        public static Color SamplePixel( Ray R, CollisionResult Res )
        {
            if ( R.Depth > RayCaster.MaxDepth )
                return Color.Black;

            if ( !Res.Hit )
                return Color.Black;

            Color Own = Res.Object.Material.Color * DiffuseLightColor( R, Res );

            return Own;
        }

        public static Color Calculate( Ray R, CollisionResult Res )
        {
            return SamplePixel( R, Res );
        }

        public static Color DiffuseLightColor( Ray R, CollisionResult Res )
        {
            Color C = Color.Black;
            foreach ( Light L in RayCaster.Lights )
            {
                bool Shadowed = IsInShadow( Res, L );

                Vector3 LightDirection = L.Position - Res.Position;
                float Length = LightDirection.Length;
                float DistDiv = ( L.FallOffDistance - Length ) / L.FallOffDistance;
                if ( DistDiv < 0 )
                    break;

                if ( DistDiv > 1f )
                    DistDiv = 1f;

                LightDirection /= Length;
                float Intensity = DistDiv * Math.Max( 0, Res.Normal.Dot( LightDirection ) ) * L.Intensity;
                float ShadowMul = ShadowMultiplier( Shadowed );

                C += L.DiffuseColor * ( ShadowMul * Intensity + L.AmbientIntensity );
            }

            return C;
        }

        public static Color Radiance( Ray R )
        {
            if ( R.Depth > RayCaster.MaxDepth )
                return Color.Black;

            CollisionResult Res = RayCaster.Trace( R );
            if ( !Res.Hit )
                return Color.Black;

            Color Rad = Res.Object.Material.Radiance;
            if ( Rad.R >= 1f || Rad.G >= 1f || Rad.B >= 1f )
                return Rad;

            Color C = new Color( 0, 0, 0 );
            for ( int Q = 0; Q < RayCaster.Samples; Q++ )
            {
                Ray New = new Ray
                {
                    Depth = R.Depth + 1,
                    Direction = Utilities.RandomCosineDirectionInSameDirection( Res.Normal ),
                    Start = Res.Position + Res.Normal * ShadowBias
                };

                // Compute the BRDF for this ray (assuming Lambertian reflection)
                float cos_theta = Math.Max( 0, New.Direction.Dot( Res.Normal ) );
                float BDRF = 2 * cos_theta;
                C += Rad + Res.Object.Material.Color * ( Radiance( New ) * BDRF * ( float ) ( 1f / Math.PI ) );
            }

            return C;
        }

        public static Color Radiance2( Ray R )
        {
            if ( R.Depth > RayCaster.MaxDepth )
                return Color.Black;

            Color C = Color.Black;

            CollisionResult Res = RayCaster.Trace( R );
            if ( !Res.Hit )
                return C;

            Color Rad = Res.Object.Material.Radiance;
            if ( Rad.R >= 1f || Rad.G >= 1f || Rad.B >= 1f )
                return Rad;

            R = new Ray
            {
                Depth = R.Depth + 1,
                Direction = Utilities.RandomCosineDirectionInSameDirection( Res.Normal ),
                Start = Res.Position + Res.Normal * ShadowBias
            };

            // Compute the BRDF for this ray (assuming Lambertian reflection)
            float cos_theta = Math.Max( 0, R.Direction.Dot( Res.Normal ) );
            float BDRF = 2 * cos_theta;

            return Rad + Res.Object.Material.Color * ( Radiance2( R ) * BDRF );
        }
    }
}
