﻿using System;

namespace Tracer.Classes.Util
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

        public Vector3( float X, float Y, float Z )
        {
            this.X = X;
            this.Y = Y;
            this.Z = Z;
        }

        /// <summary>
        /// Normalizes this vector.
        /// </summary>
        public void Normalize( )
        {
            float L = this.Length;
            this.X /= L;
            this.Y /= L;
            this.Z /= L;
        }

        /// <summary>
        /// Returns a normalized copy of this vector.
        /// </summary>
        /// <returns>A normalized copy of this vector.</returns>
        public Vector3 Normalized( )
        {
            float L = this.Length;

            return new Vector3( this.X, this.Y, this.Z ) / L;
        }

        /// <summary>
        /// Returns the dot product between this vector and a given vector.
        /// </summary>
        /// <param name="V2">The vector to compute the dot product with.</param>
        /// <returns>The dot product between the two vectors.</returns>
        public float Dot( Vector3 V2 )
        {
            return this.X * V2.X + this.Y * V2.Y + this.Z * V2.Z;
        }

        /// <summary>
        /// Returns the cross product between this vector and a given vector.
        /// </summary>
        /// <param name="V2">The vector to compute the cross product with.</param>
        /// <returns>The cross product between the two vectors.</returns>
        public Vector3 Cross( Vector3 V2 )
        {
            float X = this.Y * V2.Z - this.Z * V2.Y;
            float Y = this.Z * V2.X - this.X * V2.Z;
            float Z = this.X * V2.Y - this.Y * V2.X;

            return new Vector3( X, Y, Z );
        }

        public override string ToString( )
        {
            return string.Format( "X: {0}, Y: {1}, Z: {2}", X, Y, Z );
        }

        #region Operators

        /// <summary>
        /// Divides a vector by a float.
        /// </summary>
        /// <param name="V">The vector to divide.</param>
        /// <param name="Divider">The float to divide with.</param>
        /// <returns>A vector where all coordinates are divided by the divider.</returns>
        public static Vector3 operator /( Vector3 V, float Divider )
        {
            return new Vector3( V.X / Divider, V.Y / Divider, V.Z / Divider );
        }

        /// <summary>
        /// Multiplies a vector by a float.
        /// </summary>
        /// <param name="V">The vector to multiply.</param>
        /// <param name="Multiplier">The float to multiply with.</param>
        /// <returns>A vector where all coordinates are multiplied by the multiplier.</returns>
        public static Vector3 operator *( Vector3 V, float Multiplier )
        {
            return new Vector3( V.X * Multiplier, V.Y * Multiplier, V.Z * Multiplier );
        }

        /// <summary>
        /// Multiplies a vector by a float.
        /// </summary>
        /// <param name="V">The vector to multiply.</param>
        /// <param name="Multiplier">The float to multiply with.</param>
        /// <returns>A vector where all coordinates are multiplied by the multiplier.</returns>
        public static Vector3 operator *( float Multiplier, Vector3 V )
        {
            return new Vector3( V.X * Multiplier, V.Y * Multiplier, V.Z * Multiplier );
        }

        /// <summary>
        /// Adds together two vectors.
        /// </summary>
        /// <param name="V1"></param>
        /// <param name="V2"></param>
        /// <returns></returns>
        public static Vector3 operator +( Vector3 V1, Vector3 V2 )
        {
            return new Vector3( V1.X + V2.X, V1.Y + V2.Y, V1.Z + V2.Z );
        }

        /// <summary>
        /// Subtracts a vector from another vector.
        /// </summary>
        /// <param name="V1">The vector to subtract from.</param>
        /// <param name="V2">The vector to subtract.</param>
        /// <returns></returns>
        public static Vector3 operator -( Vector3 V1, Vector3 V2 )
        {
            return new Vector3( V1.X - V2.X, V1.Y - V2.Y, V1.Z - V2.Z );
        }

        /// <summary>
        /// Negates a vector.
        /// </summary>
        /// <param name="V"></param>
        /// <returns></returns>
        public static Vector3 operator -( Vector3 V )
        {
            return new Vector3( -V.X, -V.Y, -V.Z );
        }

        #endregion
    }
}