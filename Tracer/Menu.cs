using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Tracer.Classes;
using Tracer.Classes.Objects;
using Tracer.Properties;
using Color = Tracer.Classes.Util.Color;

namespace Tracer
{
    public partial class Menu : Form
    {
        public static Camera Cam;
        private Color [ , ] Img;
        private int Done;
        private int Max;
        private bool Rendering;

        public Menu( )
        {
            InitializeComponent( );
        }

        protected override void OnLoad( EventArgs e )
        {
            base.OnLoad( e );

            Cam = new Camera( 750, 750, 80 );

            RayCaster.Objects.Add( new Sphere( Cam.Angle.Forward * 50, 20 ) );
            RayCaster.Objects.Add( new Sphere( Cam.Angle.Forward * 40 + Cam.Angle.Up * 5 + Cam.Angle.Right * 5, 10 )
            {
                Material = new Material { Color = new Color( 255, 0, 0 ) }
            } );
            RayCaster.Objects.Add( new Sphere( Cam.Angle.Forward * 42 + Cam.Angle.Up * 25, 3f )
            {
                Material = new Material { Color = new Color( 0, 255, 0 ) }
            } );

            RayCaster.Lights.Add( new Light
            {
                DiffuseColor = new Color( 255, 255, 255 ),
                Position = Cam.Angle.Forward * 48 + Cam.Angle.Up * 38,
                FallOffDistance = 150,
                AmbientIntensity = .05f,
                Intensity = 1f
            } );
        }

        private void RenderImage( int W, int H )
        {
            if ( Rendering )
                return;

            Done = 0;
            Status_Label.Text = Resources.Status_Rendering;
            Img = new Color[ W, H ];
            Rendering = true;
            int Percentage = ( int ) ( Max * 0.05f );

            BackgroundWorker Worker = new BackgroundWorker( );
            Worker.DoWork += ( Obj, Args ) =>
            {
                int D = 0;
                Parallel.For( 0, Max, Var =>
                {
                    int X = Var % W;
                    int Y = Var / H;

                    Img[ X, Y ] = RayCaster.Cast( Cam.GetRay( X, Y ) );
                    Interlocked.Increment( ref D );
                    if ( D % Percentage != 0 ) return;

                    Done = D;
                    this.Invoke( ( MethodInvoker )( ( ) => this.Status_Progress.Value = Done ) );
                } );
                this.Invoke( ( MethodInvoker )( ( ) => this.Status_Label.Text = Resources.Statuses_Drawing ) );
            };

            Worker.RunWorkerCompleted += ( Obj, Args ) => new Thread( ( ) =>
            {
                foreach ( Bitmap B in DrawRenderedImage( ) )
                    this.Invoke( ( MethodInvoker ) ( ( ) => pictureBox1.Image = B ) );
                Rendering = false;
            } ).Start( );

            Worker.RunWorkerAsync( );
        }

        private IEnumerable<Bitmap> DrawRenderedImage( )
        {
            Status_Label.Text = Resources.Statuses_Drawing;
            Bitmap B = new Bitmap( Img.GetLength( 0 ), Img.GetLength( 1 ) );
            Done = 0;

            int Percentage = ( int ) ( Max * 0.01f );

            for ( int X = 0; X < Cam.Resolution.X; X++ )
            {
                for ( int Y = 0; Y < Cam.Resolution.Y; Y++ )
                {
                    B.SetPixel( X, Y, Img[ X, Y ].DrawingColor );
                    if ( Done++ % Percentage == 0 )
                    {
                        this.Invoke( ( MethodInvoker ) ( ( ) =>
                        {
                            this.Status_Progress.Value = Done;
                        } ) );

                        yield return B.Clone( new Rectangle( 0, 0, B.Width, B.Height ), B.PixelFormat );
                    }
                }
            }

            yield return B;
        }

        private void Button_Render_Click( object sender, EventArgs e )
        {
            Max = ( int ) ( Cam.Resolution.X * Cam.Resolution.Y );
            this.Status_Progress.Maximum = Max;

            int W = ( int ) Cam.Resolution.X;
            int H = ( int ) Cam.Resolution.Y;

            RenderImage( W, H );
        }
    }
}
