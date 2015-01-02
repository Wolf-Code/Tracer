using System;
using System.Windows.Forms;
using Tracer.Classes;
using Tracer.Classes.Objects;
using Tracer.Properties;

namespace Tracer
{
    public static class Renderer
    {
        public static Camera Cam;
        private static Menu Menu;
        public static Scene Scene;

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

            Menu.Settings_Samples.Value = Settings.Default.Render_Samples;
            Menu.Settings_Depth.Value = Settings.Default.Render_MaxDepth;

            Scene = CUDAInterface.DefaultScene;

            CUDAInterface.OnProgress += CudaTestOnOnProgress;
            CUDAInterface.OnFinished += CUDATest_OnFinished;

            try
            {
                Scene = Content.Load<Scene>( Settings.Default.Location_LastScene );
            }
            catch
            {
                Scene = CUDAInterface.DefaultScene;
            }

            Menu.SceneProperties.SelectedObject = Scene;
        }

        private static void CUDATest_OnFinished( object sender, CUDAFinishedEventArgs e )
        {
            Menu.Invoke( ( MethodInvoker ) ( ( ) =>
            {
                Menu.RenderImage.Image = e.Image;
                Menu.Status_Progress.Value = Menu.Status_Progress.Maximum;
                OnRenderingEnded( );

                MessageBox.Show(
                    string.Format( "Render time: {0}. Average area time: {1}", e.Time, e.AverageProgressTime ),
                    Resources.Status_Done );
            } ) );
        }

        private static void CudaTestOnOnProgress( object Sender, CUDAProgressEventArgs E )
        {
            Menu.Invoke( ( MethodInvoker )( ( ) =>
            {
                Menu.Status_Progress.Value = ( int )( E.TotalProgress * Menu.Status_Progress.Maximum );
                if ( E.TotalProgress < 1 )
                {
                    Menu.Status_Label.Text =
                        new TimeSpan(
                            ( long ) ( ( ( 1.0 - E.TotalProgress ) / E.Progress ) * E.AverageProgressTime.Ticks ) )
                            .ToString( );
                }
                else
                {
                    Menu.Status_Label.Text = Resources.Statuses_Drawing;
                }
            } ) );
        }

        private static void OnRenderingEnded( )
        {
            Menu.Button_Render.Text = Resources.Line_Render;
            Rendering = false;
        }

        public static void CancelRendering( )
        {
            if ( !Rendering )
                return;

            CUDAInterface.Cancel( );
            OnRenderingEnded( );
        }

        public static void RenderImage( )
        {
            if ( Rendering )
                return;

            Rendering = true;
            CUDAInterface.Run( );
            Menu.Button_Render.Text = Resources.Line_Cancel;
        }
    }
}
