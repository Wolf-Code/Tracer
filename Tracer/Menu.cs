using System;
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
        private Thread RenderThread;
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

            Cam = new Camera( 1000, 1000, 80 );

            Sphere S = new Sphere( Cam.Angle.Forward * 50, 20 );
            RayCaster.Objects.Add( S );
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
                    if ( D % Percentage == 0 )
                    {
                        Done = D;
                        this.Invoke( ( MethodInvoker )( ( ) => this.Status_Progress.Value = Done ) );
                    }
                } );
                this.Invoke( ( MethodInvoker )( ( ) => this.Status_Label.Text = Resources.Statuses_Drawing ) );
            };

            Worker.RunWorkerCompleted += ( Obj, Args ) => new Thread( ( ) =>
            {
                Bitmap B = DrawRenderedImage( );
                this.Invoke( ( MethodInvoker ) ( ( ) => pictureBox1.Image = B ) );
                Rendering = false;
            } ).Start( );

            Worker.RunWorkerAsync( );
        }

        private Bitmap DrawRenderedImage( )
        {
            Status_Label.Text = Resources.Statuses_Drawing;
            Bitmap B = new Bitmap( Img.GetLength( 0 ), Img.GetLength( 1 ) );
            Done = 0;
            using ( Graphics G = Graphics.FromImage( B ) )
            {
                for ( int X = 0; X < Cam.Resolution.X; X++ )
                {
                    for ( int Y = 0; Y < Cam.Resolution.Y; Y++ )
                    {
                        G.DrawRectangle( new Pen( Img[ X, Y ].DrawingColor ), X, Y, 1, 1 );
                        Done++;
                        this.Invoke( ( MethodInvoker ) ( ( ) => this.Status_Progress.Value = Done ) );
                    }
                }
            }

            return B;
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
