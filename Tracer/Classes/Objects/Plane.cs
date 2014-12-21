using Tracer.Classes.Util;

namespace Tracer.Classes.Objects
{
    /// <summary>
    /// A sphere object.
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
        public float Distance;

        public Plane( Vector3 Normal, float Distance )
        {
            this.Normal = Normal;
            this.Distance = Distance;
        }

        public override CollisionResult CheckCollision( Ray R )
        {
            CollisionResult Result = new CollisionResult( );

            return Result;
        }
    }
}
