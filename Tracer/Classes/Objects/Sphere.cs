using System;
using Tracer.Classes.Util;
using Tracer.CUDA;

namespace Tracer.Classes.Objects
{
    /// <summary>
    /// A sphere object.
    /// </summary>
    public class Sphere : GraphicsObject
    {
        /// <summary>
        /// The sphere's position.
        /// </summary>
        public Vector3 Center { set; get; }

        /// <summary>
        /// The sphere's radius.
        /// </summary>
        public float Radius { set; get; }

        public Sphere( Vector3 Position, float Radius )
        {
            this.Center = Position;
            this.Radius = Radius;
            this.Name = "Sphere";
        }

        public override CollisionResult CheckCollision( Ray R )
        {
            CollisionResult Result = new CollisionResult( );

            #region Distance
            // Created after http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
            float A = R.Direction.Dot( R.Direction );
            float B = 2 * R.Direction.Dot( R.Start - this.Center );
            float C = ( R.Start - this.Center ).Dot( R.Start - this.Center ) - ( this.Radius * this.Radius );

            float Discriminant = B * B - 4 * A * C;
            if ( Discriminant < 0 )
                return Result;

            float DiscriminantSqrt = ( float )Math.Sqrt( Discriminant );
            float Q;
            if ( B < 0 )
                Q = ( -B - DiscriminantSqrt ) / 2f;
            else
                Q = ( -B + DiscriminantSqrt ) / 2f;

            float T0 = Q / A;
            float T1 = C / Q;

            if ( T0 > T1 )
            {
                float TempT0 = T0;
                T0 = T1;
                T1 = TempT0;
            }

            // Sphere is behind the ray's start position.
            if ( T1 < 0 )
                return Result;

            Result.Distance = T0 < 0 ? T1 : T0;
            Result.Hit = true;
            #endregion

            if ( Result.Hit )
            {
                Vector3 Position = R.Start + R.Direction * Result.Distance;
                //Vector3 North = new Vector3( 0, 1, 0 );
                //Vector3 East = new Vector3( -1, 0, 0 );
                Vector3 Normal = ( Position - this.Center ).Normalized( );

                //float U = ( 0.5f + ( float )( Math.Atan2( -Normal.Z, -Normal.X ) / ( 2 * Math.PI ) ) ) * this.Material.TextureUScale;
                //float V = ( 0.5f - ( float )( Math.Asin( -Normal.Y ) / Math.PI ) ) * this.Material.TextureVScale;

                /*if ( this.Material.HasNormalMap )
                {
                    Vector3 Tangent;
                    if ( Normal != new Vector3( 1, 0, 0 ) )
                        Tangent = Normal.Cross( new Vector3( 1, 0, 0 ) );
                    else
                        Tangent = Normal.Cross( new Vector3( 0, 0, 1 ) );

                    Tangent.Normalize( );

                    Vector3 BiNormal = Normal.Cross( Tangent );

                    Vector3 TangentSpaceNormal = this.Material.GetNormal( U, V );
                    Normal = Tangent * TangentSpaceNormal.X + BiNormal * TangentSpaceNormal.Y + Normal * TangentSpaceNormal.Z;
                    Normal.Normalize( );
                }

                System.Drawing.PointF TexCoords = new System.Drawing.PointF( U, V );*/
                //Result = new Tracing.CollisionResult( true, Position, Normal, Distance, this, TexCoords );
                Result.Position = Position;
                Result.Normal = Normal;
                Result.Object = this;
            }

            return Result;
        }

        public CUDASphereObject ToCUDASphere( )
        {
            return new CUDASphereObject
            {
                Position = this.Center.ToFloat3( ),
                Radius = this.Radius
            };
        }
    }
}
