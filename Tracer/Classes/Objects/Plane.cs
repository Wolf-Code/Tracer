using System.ComponentModel;
using Tracer.Classes.Util;
using Tracer.CUDA;

namespace Tracer.Classes.Objects
{
    /// <summary>
    /// A plane object.
    /// </summary>
    public class Plane : GraphicsObject
    {
        /// <summary>
        /// The plane's normal.
        /// </summary>
        public Vector3 Normal { set; get; }

        /// <summary>
        /// The plane's distance from the center, along its normal.
        /// </summary>
        [Description( "The offset from the center, in its normal's direction." )]
        public float Offset { set; get; }

        public Plane( Vector3 Normal, float Offset )
        {
            this.Normal = Normal;
            this.Offset = Offset;
            this.Name = "Plane";
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

        public CUDAPlane ToCUDAPlane( )
        {
            return new CUDAPlane
            {
                Normal = this.Normal.ToFloat3( ),
                Offset = this.Offset
            };
        }
    }
}
