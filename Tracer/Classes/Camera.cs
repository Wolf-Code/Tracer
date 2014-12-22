using System;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;

namespace Tracer.Classes
{
    /// <summary>
    /// A camera class.
    /// </summary>
    public class Camera
    {
        /// <summary>
        /// The position of the camera.
        /// </summary>
        public Vector3 Position;

        /// <summary>
        /// The angle of the camera.
        /// </summary>
        public Angle Angle;

        /// <summary>
        /// The resolution of the camera.
        /// </summary>
        public Vector2 Resolution;

        private float Fov;

        /// <summary>
        /// The field of view of the camera.
        /// </summary>
        public float FOV
        {
            set
            {
                Fov = value;
                A = 0.5f / ( float ) Math.Tan( MathHelper.ToRadians( FOV / 2 ) );
            }
            get { return Fov; }
        }

        private float A;

        public Camera( int Width, int Height, float FOV )
        {
            this.Angle = new Angle( );
            this.Position = new Vector3( 0, 0, 0 );
            this.Resolution = new Vector2( Width, Height );
            this.FOV = FOV;
        }

        /// <summary>
        /// Creates a ray for a given pixel.
        /// </summary>
        /// <param name="PixelX">The horizontal pixel.</param>
        /// <param name="PixelY">The vertical pixel.</param>
        /// <returns>The ray belonging to this pixel.</returns>
        public Ray GetRay( int PixelX, int PixelY )
        {
            Ray R = new Ray { Start = this.Position };
            Vector3 Dir = this.Angle.Forward * A + 
                          this.Angle.Right * ( PixelX / Resolution.X - 0.5f ) - 
                          this.Angle.Up * ( PixelY / Resolution.Y - 0.5f );

            R.Direction = Dir.Normalized(  );

            return R;
        }
    }
}
