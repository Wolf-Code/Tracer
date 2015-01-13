using System;
using System.ComponentModel;
using Tracer.Classes.Util;
using Tracer.Structs.CUDA;
using Tracer.Utilities;

namespace Tracer.Classes.SceneObjects
{
    /// <summary>
    /// A camera class.
    /// </summary>
    [Serializable]
    [TypeConverter( typeof ( ExpandableObjectConverter ) )]
    [Description( "Contains data about how to render the scene." )]
    public class Camera
    {
        /// <summary>
        /// The position of the camera.
        /// </summary>
        [Description( "The position from which is being rendered." )]
        public Vector3 Position { set; get; }

        /// <summary>
        /// The angle of the camera.
        /// </summary>
        [Description( "The camera's angle, which changes the direction the camera is pointing at." )]
        public Angle Angle { set; get; }

        /// <summary>
        /// The resolution of the camera.
        /// </summary>
        [Description( "The resolution with which to render." )]
        public Vector2 Resolution { set; get; }

        private float Fov;

        /// <summary>
        /// The field of view of the camera.
        /// </summary>
        [DisplayName( @"Field of view" )]
        [Description( "The amount of degrees the camera can see in front of it." )]
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

        public Camera( )
        {
            this.Resolution = new Vector2( 1920, 1080 );
            this.FOV = 90;
            this.Angle = new Angle( );
            this.Position = new Vector3( 0, 0, 0 );
        }

        public Camera( int Width, int Height, float FOV ) : this( )
        {
            this.Resolution = new Vector2( Width, Height );
            this.FOV = FOV;
        }

        public CUDACamData ToCamData( )
        {
            CUDACamData Data = new CUDACamData
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

        public override string ToString( )
        {
            return this.Resolution.ToString( );
        }
    }
}