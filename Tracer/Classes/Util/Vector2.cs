using System;

namespace Tracer.Classes.Util
{
    /// <summary>
    /// A class representing a 2D vector.
    /// </summary>
    public class Vector2
    {
        /// <summary>
        /// The X-coordinate.
        /// </summary>
        public float X;

        /// <summary>
        /// The Y-coordinate.
        /// </summary>
        public float Y;

        /// <summary>
        /// The squared length of this vector.
        /// </summary>
        public float LengthSquared
        {
            get { return X * X + Y * Y; }
        }

        /// <summary>
        /// The length of this vector.
        /// </summary>
        public float Length
        {
            get { return ( float )Math.Sqrt( LengthSquared ); }
        }

        /// <summary>
        /// Creates a new vector2.
        /// </summary>
        /// <param name="X">The X-coordinate.</param>
        /// <param name="Y">The Y-coordinate.</param>
        public Vector2( float X, float Y )
        {
            this.X = X;
            this.Y = Y;
        }
    }
}
