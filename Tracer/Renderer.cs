using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Threading;
using System.Windows.Forms;
using Tracer.Classes;
using Tracer.Properties;
using Color = Tracer.Classes.Util.Color;

namespace Tracer
{
    public static class Renderer
    {
        public static Camera Cam;
        private static Menu Menu;

        private static Color[ , ] Img;
        private static int Done;
        private static int Max;
        public static bool Rendering { private set; get; }

        public static void Initialize( Menu M )
        {
            Cam = new Camera( Settings.Default.Render_Resolution_Width, 
                Settings.Default.Render_Resolution_Height,
                Settings.Default.Render_FOV );
            Menu = M;

            Menu.Settings_Resolution_Width.Value = Settings.Default.Render_Resolution_Width;
            Menu.Settings_Resolution_Height.Value = Settings.Default.Render_Resolution_Height;
            Menu.Settings_FOV.Value = ( decimal ) Settings.Default.Render_FOV;

            if ( Settings.Default.Image_Folder.Length == 0 )
            {
                Settings.Default[ "Image_Folder" ] = Environment.CurrentDirectory + "\\Images";
                Settings.Default.Save( );
            }

            Menu.Settings_ImageFolder.Text = Settings.Default.Image_Folder;
        }

        public static void RenderImage( )
        {
            if ( Rendering )
                return;

            int W = ( int )Cam.Resolution.X;
            int H = ( int )Cam.Resolution.Y;

            Max = W * H;
            Done = 0;
            Menu.Status_Progress.Maximum = Max;
            Menu.Status_Label.Text = Resources.Status_Rendering;
            Img = new Color[ W, H ];
            Rendering = true;
            int Percentage = ( int )( Max * 0.03f );

            Thread T = new Thread( ( ) =>
            {
                
            } );
            T.Start( );

            BackgroundWorker Worker = new BackgroundWorker( );
            Worker.DoWork += ( Sender, Args ) =>
            {
                for ( int X = 0; X < W; X++ )
                    for ( int Y = 0; Y < H; Y++ )
                    {
                        Img[ X, Y ] = RayCaster.Cast( Cam.GetRay( X, Y ) );
                        if( Done++ % Percentage == 0 )
                            Menu.Invoke( ( MethodInvoker )( ( ) => Menu.Status_Progress.Value = Done ) );
                    }
            };

            Worker.RunWorkerCompleted += ( Sender, Args ) =>
            {
                foreach ( Bitmap B in DrawRenderedImage( ) )
                    Menu.Invoke( ( MethodInvoker )( ( ) => Menu.RenderImage.Image = B ) );
                Rendering = false;
            };

            Worker.RunWorkerAsync( );
        }

        private static IEnumerable<Bitmap> DrawRenderedImage( )
        {
            Menu.Status_Label.Text = Resources.Statuses_Drawing;
            Bitmap B = new Bitmap( Img.GetLength( 0 ), Img.GetLength( 1 ) );
            Done = 0;

            int Percentage = ( int )( Max * 0.01f );

            for ( int X = 0; X < Cam.Resolution.X; X++ )
            {
                for ( int Y = 0; Y < Cam.Resolution.Y; Y++ )
                {
                    B.SetPixel( X, Y, Img[ X, Y ].DrawingColor );
                    if ( Done++ % Percentage != 0 ) continue;

                    Menu.Invoke( ( MethodInvoker )( ( ) =>
                    {
                        Menu.Status_Progress.Value = Done;
                    } ) );

                    yield return B.Clone( new Rectangle( 0, 0, B.Width, B.Height ), B.PixelFormat );
                }
            }

            yield return B;
        }
    }
}
