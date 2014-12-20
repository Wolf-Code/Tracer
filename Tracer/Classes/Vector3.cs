using System;

namespace Tracer.Classes
{
    /// <summary>
    /// A class representing a 3D vector.
    /// </summary>
    public class Vector3
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
        /// The Z-coordinate.
        /// </summary>
        public float Z;

        /// <summary>
        /// The squared length of this vector.
        /// </summary>
        public float LengthSquared
        {
            get { return X * X + Y * Y + Z * Z; }
        }

        /// <summary>
        /// The length of this vector.
        /// </summary>
        public float Length
        {
            get { return ( float )Math.Sqrt( LengthSquared ); }
        }
    }
}
