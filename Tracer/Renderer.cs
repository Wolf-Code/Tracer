using System;
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

            foreach ( IDevice Dev in RenderInstance.GetDevices( ) )
                Menu.Settings_Devices.Items.Add( Dev );

            Menu.Settings_Devices.SelectedIndex = 0;
        }

        private static void RendererOnFinished(object sender, RendererFinishedEventArgs e)
        {
            Menu.Perform( ( ) =>
            {
                Menu.RenderImage.Image = e.Image;
                Menu.Status_Progress.Value = Menu.Status_Progress.Maximum;
                OnRenderingEnded( );
            } );

            Output.WriteLine( string.Format( "Render time: {0}. Average area time: {1}", e.Time, e.AverageProgressTime ) );
        }

        private static void RendererOnProgress(object Sender, RendererProgressEventArgs E)
        {
            Menu.Perform( ( ) =>
            {
                Menu.Status_Progress.Value = ( int ) ( E.TotalProgress * Menu.Status_Progress.Maximum );
                if ( E.TotalProgress < 1 )
                {
                    Menu.Status_Label.Text =
                        new TimeSpan(
                            ( long ) ( ( ( 1.0 - E.TotalProgress ) / E.Progress ) * E.AverageProgressTime.Ticks ) )
                            .ToString( );
                    Output.WriteLine( "Rendered image area {0} of {1} in {2}", ( int )( E.TotalProgress / E.Progress ), ( int )( 1f / E.Progress ), E.ProgressTime );
                }
                else
                {
                    Menu.Status_Label.Text = Resources.Statuses_Drawing;
                }
            } );
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
            Output.WriteLine( "Canceling rendering. Finishing area.." );
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
