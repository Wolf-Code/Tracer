using Tracer.Classes.Objects;
using Tracer.Classes.Util;

namespace Tracer.Classes
{
    /// <summary>
    /// The result of a collision, containing relevant information.
    /// </summary>
    public struct CollisionResult
    {
        /// <summary>
        /// Was there an actual collision?
        /// </summary>
        public bool Hit;

        /// <summary>
        /// The position of the collision.
        /// </summary>
        public Vector3 Position;

        /// <summary>
        /// The normal of the surface where the collision occured.
        /// </summary>
        public Vector3 Normal;

        /// <summary>
        /// The distance from the start of the ray to the collision.
        /// </summary>
        public float Distance;

        /// <summary>
        /// The object with which there was a collision.
        /// </summary>
        public GraphicsObject Object;
    }
}
