using System;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;
using Tracer.CUDA;

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

        public CUDACamData ToCamData( )
        {
            CUDACamData Data = new CUDACamData( )
            {
                A = this.A,
                Forward = this.Angle.Forward.ToFloat3( ),
                Height = this.Resolution.Y,
                Position = this.Position.ToFloat3( ),
                Right = this.Angle.Right.ToFloat3( ),
                Up = this.Angle.Up.ToFloat3( ),
                Width = this.Resolution.X
            };

            return Data;
        }
    }
}
