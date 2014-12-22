using Tracer.Classes.Util;

namespace Tracer.Classes.Objects
{
    /// <summary>
    /// A plane object.
    /// </summary>
    class Plane : GraphicsObject
    {
        /// <summary>
        /// The plane's normal.
        /// </summary>
        public Vector3 Normal;

        /// <summary>
        /// The plane's distance from the center, along its normal.
        /// </summary>
        public float Offset;

        public Plane( Vector3 Normal, float Offset )
        {
            this.Normal = Normal;
            this.Offset = Offset;
        }

        public override CollisionResult CheckCollision( Ray R )
        {
            CollisionResult Result = new CollisionResult( );

            float Div = this.Normal.Dot( R.Direction );
            // Plane is orthogonal, no collision.
            if ( Div == 0 )
                return Result;

            float Distance = -( this.Normal.Dot( R.Start ) + this.Offset ) / Div;

            if ( Distance < 0 )
                return Result;

            Result.Hit = true;
            Result.Object = this;
            Result.Distance = Distance;
            Result.Normal = this.Normal;
            Result.Position = R.Start + R.Direction * Distance;
            return Result;
        }
    }
}
