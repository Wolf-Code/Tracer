using System;
using System.Windows.Forms;
using Tracer.Classes;
using Tracer.Classes.Objects;
using Tracer.Properties;
using Tracer.Renderers;

namespace Tracer
{
    public static class Renderer
    {
        private static Menu Menu;
        public static Scene Scene;
        private static IRenderer RenderInstance;

        public static bool Rendering { private set; get; }

        public static void Initialize( Menu M )
        {
            Menu = M;

            Menu.Settings_Samples.Value = Settings.Default.Render_Samples;
            Menu.Settings_Depth.Value = Settings.Default.Render_MaxDepth;

            Scene = Scene.Default;

            try
            {
                Scene = Content.Load<Scene>( Settings.Default.Location_LastScene );
            }
            catch
            {
                Scene = Scene.Default;
            }

            Menu.SceneProperties.SelectedObject = Scene;

            RenderInstance = new CUDARenderer( );

            RenderInstance.OnProgress += RendererOnProgress;
            RenderInstance.OnFinished += RendererOnFinished;
        }

        private static void RendererOnFinished(object sender, RendererFinishedEventArgs e)
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

        private static void RendererOnProgress(object Sender, RendererProgressEventArgs E)
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

            RenderInstance.Cancel( );
            OnRenderingEnded( );
        }

        public static void RenderImage( )
        {
            if ( Rendering )
                return;

            Rendering = true;
            RenderInstance.Run( );
            Menu.Button_Render.Text = Resources.Line_Cancel;
        }
    }
}
