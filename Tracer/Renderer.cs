using System;
using System.Drawing;
using System.Windows.Forms;
using Tracer.Classes.SceneObjects;
using Tracer.Interfaces;
using Tracer.Properties;
using Tracer.Renderers;
using Tracer.TracerEventArgs;
using Tracer.Utilities;

namespace Tracer
{
    public static class Renderer
    {
        private static Menu Menu;
        public static Scene Scene;
        private static IRenderer RenderInstance;
        private static bool IsWindowActive;

        public static bool Rendering { private set; get; }

        public static void Initialize( Menu M )
        {
            Menu = M;
            Menu.Activated += ( obj, snd ) => IsWindowActive = true;
            Menu.Deactivate += ( obj, snd ) => IsWindowActive = false;


            Menu.Settings_Samples.Value = Settings.Default.Render_Samples;
            Menu.Settings_Depth.Value = Settings.Default.Render_MaxDepth;
            Menu.Settings_AreaDivider.Value = Settings.Default.Render_AreaDivider;

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

            RenderInstance.OnFinished += RendererOnFinished;
            RenderInstance.OnSampleFinished += RenderInstance_OnSampleFinished;

            foreach ( IDevice Dev in RenderInstance.GetDevices( ) )
                Menu.Settings_Devices.Items.Add( Dev );

            Menu.Settings_Devices.SelectedIndex = 0;
        }

        private static void RenderInstance_OnSampleFinished( object Sender, RenderSampleEventArgs E )
        {
            Menu.Perform( ( ) =>
            {
                Menu.Status_Progress.Value = ( int ) ( E.Progress * Menu.Status_Progress.Maximum );

                Menu.RenderImage.Image = new Bitmap( E.Image );

                if ( E.Progress < 1 )
                {
                    double Sample = Math.Round( E.TotalSamples * E.Progress );
                    float RemainingProgress = ( 1.0f - E.Progress );
                    long EstimatedTimeLeft = ( long ) ( RemainingProgress * E.TotalSamples * E.AverageSampleTime.Ticks );
                    Menu.Status_Label.Text = new TimeSpan( EstimatedTimeLeft ).ToString( );

                    Output.WriteLine( "Rendered sample {0} of {1} in {2}", Sample,
                        E.TotalSamples, E.Time );
                }
                else
                {
                    Menu.Status_Label.Text = Resources.Statuses_Drawing;
                }
            } );
        }

        private static void RendererOnFinished( object sender, RendererFinishedEventArgs e )
        {
            Menu.Perform( ( ) =>
            {
                Menu.RenderImage.Image = e.Image;
                Menu.Status_Progress.Value = Menu.Status_Progress.Maximum;
                OnRenderingEnded( );
            } );

            Output.WriteLine( string.Format( "Render time: {0}. Average area time: {1}", e.Time, e.AverageProgressTime ) );

            if ( !IsWindowActive )
                Menu.Notifier.ShowBalloonTip( 500, "Rendering finished", "Rendering took " + e.Time,
                    ToolTipIcon.Info );
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

        public static void NextArea( )
        {
            if ( !Rendering )
                return;

            RenderInstance.NextArea( );
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