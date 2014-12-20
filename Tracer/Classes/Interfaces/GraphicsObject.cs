using Tracer.Classes.Objects;

namespace Tracer.Classes.Interfaces
{
    /// <summary>
    /// An interface for all objects which we can represent in the raytraced image.
    /// </summary>
    public interface IGraphicsObject
    {
        /// <summary>
        /// Checks for a collision with a ray.
        /// </summary>
        /// <param name="R">The ray to check for.</param>
        /// <returns>A <see cref="CollisionResult"/> containing all collision information.</returns>
        CollisionResult CheckCollision( Ray R );
    }
}
