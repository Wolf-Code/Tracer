
namespace Tracer.Classes.Objects
{
    /// <summary>
    /// An abstract class for all objects which we can represent in the raytraced image.
    /// </summary>
    public abstract class GraphicsObject
    {
        /// <summary>
        /// The material of the object.
        /// </summary>
        public Material Material { set; get; }

        /// <summary>
        /// The name to identify the object with.
        /// </summary>
        public string Name { set; get; }

        protected GraphicsObject( )
        {
            this.Material = new Material( );
        }

        /// <summary>
        /// Checks for a collision with a ray.
        /// </summary>
        /// <param name="R">The ray to check for.</param>
        /// <returns>A <see cref="CollisionResult"/> containing all collision information.</returns>
        public abstract CollisionResult CheckCollision( Ray R );

        public override string ToString( )
        {
            return this.Name;
        }
    }
}
