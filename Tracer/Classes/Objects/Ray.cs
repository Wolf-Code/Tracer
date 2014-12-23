using Tracer.Classes.Util;

namespace Tracer.Classes.Objects
{
    /// <summary>
    /// A class representing a ray in 3D space.
    /// </summary>
    public struct Ray
    {
        /// <summary>
        /// The start of the ray.
        /// </summary>
        public Vector3 Start;

        /// <summary>
        /// The normalized direction vector of the ray.
        /// </summary>
        public Vector3 Direction;



        /// <summary>
        /// The amount of times this ray has been reflected already.
        /// </summary>
        public int Depth;
    }
}
