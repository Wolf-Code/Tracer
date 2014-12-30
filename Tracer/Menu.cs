using System;
using System.Windows.Forms;
using Tracer.Classes;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;
using Tracer.Properties;
using Color = Tracer.Classes.Util.Color;

namespace Tracer
{
    public partial class Menu : Form
    {
        private int LastDone;
        private float AverageProgressPerSecond;

        public Menu( )
        {
            InitializeComponent( );
        }

        protected override void OnLoad( EventArgs e )
        {
            base.OnLoad( e );

            Renderer.Initialize( this );

            Renderer.Cam.Position = new Vector3( 0, 45, 80 );
            Renderer.Cam.Angle = new Angle { Pitch = 0, Yaw = 0, Roll = 0f };

            CUDATest.Run( );
            
            RayCaster.Objects.Add( new Sphere( new Vector3( -20, 50, -30 ), 20 )
            {
                Material = new Material
                {
                    Color = new Color( 0f, 1f, 0f )
                }
            } );

            // Light
            float LightSize = 2000;
            RayCaster.Objects.Add( new Sphere( new Vector3( 0, LightSize + 90 - .1f, 0 ), LightSize )
            {
                Material = new Material
                {
                    Color = new Color( 1f, 1f, 1f ),
                    Radiance = new Color( 1f, 1f, 1f )
                }
            } );
            
            // Floor
            RayCaster.Objects.Add( new Plane( new Vector3( 0, 1, 0 ), 0 )
            {
                Material = new Material
                {
                    Color = new Color( 1f, 1f, 1f )
                }
            } );

            // Back
            RayCaster.Objects.Add( new Plane( new Vector3( 0, 0, 1 ), 90 )
            {
                Material = new Material
                {
                    Color = new Color( 1f, 1f, 1f )
                }
            } );
            
            // Ceiling
            RayCaster.Objects.Add( new Plane( new Vector3( 0, -1, 0 ), 90 )
            {
                Material = new Material
                {
                    Color = new Color( 1f, 1f, 1f )
                }
            } );

            // Left
            RayCaster.Objects.Add( new Plane( new Vector3( 1, 0, 0 ), 90 )
            {
                Material = new Material
                {
                    Color = new Color( 1f, 0f, 0f )
                }
            } );

            // Right
            RayCaster.Objects.Add( new Plane( new Vector3( -1, 0, 0 ), 90 )
            {
                Material = new Material
                {
                    Color = new Color( 0f, 0f, 1f )
                }
            } );

            RayCaster.Lights.Add( new Light
            {
                DiffuseColor = new Color( 255, 255, 255 ),
                Position = new Vector3( 150, 70, -35 ),
                FallOffDistance = 300f,
                AmbientIntensity = .01f,
                Intensity = 1f
            } );
        }

        private void Button_Render_Click( object sender, EventArgs e )
        {
            if ( !Renderer.Rendering )
                Renderer.RenderImage( );
            else
                Renderer.CancelRendering( );
        }

        private void Settings_Resolution_Width_ValueChanged( object sender, EventArgs e )
        {
            Renderer.Cam.Resolution.X = ( float ) Settings_Resolution_Width.Value;
            Settings.Default[ "Render_Resolution_Width" ] = ( int ) Settings_Resolution_Width.Value;
            Settings.Default.Save( );
        }

        private void Settings_Resolution_Height_ValueChanged( object sender, EventArgs e )
        {
            Renderer.Cam.Resolution.Y = ( float )Settings_Resolution_Height.Value;
            Settings.Default[ "Render_Resolution_Height" ] = ( int )Settings_Resolution_Height.Value;
            Settings.Default.Save( );
        }

        private void Settings_FOV_ValueChanged( object sender, EventArgs e )
        {
            Renderer.Cam.FOV = ( float )Settings_FOV.Value;
            Settings.Default[ "Render_FOV" ] = Renderer.Cam.FOV;
            Settings.Default.Save( );
        }

        private void Settings_ImageFolder_Browse_Click( object sender, EventArgs e )
        {
            FolderBrowserDialog Dialog = new FolderBrowserDialog
            {
                SelectedPath = Settings.Default.Image_Folder,
                Description = Resources.Settings_ImageFolder_Description,
                ShowNewFolderButton = true
            };
            if ( Dialog.ShowDialog( ) != DialogResult.OK ) return;

            Settings.Default[ "Image_Folder" ] = Dialog.SelectedPath;
            Settings.Default.Save( );

            Settings_ImageFolder.Text = Settings.Default.Image_Folder;
        }

        private void ToolStrip_Button_Save_Click( object sender, EventArgs e )
        {
            if ( Renderer.Rendering ) return;
            if ( this.RenderImage.Image == null ) return;

            SaveFileDialog Dialog = new SaveFileDialog
            {
                Filter = "PNG (*.png)|*.png",
                InitialDirectory = Settings.Default.Image_Folder,
                FileName = DateTime.Now.ToShortDateString( ) + "_" + DateTime.Now.ToLongTimeString(  ).Replace( ':', '-' ) + ".png"
            };

            if ( Dialog.ShowDialog( ) == DialogResult.OK )
                this.RenderImage.Image.Save( Dialog.FileName );
        }

        private void Settings_Depth_ValueChanged( object sender, EventArgs e )
        {
            RayCaster.MaxDepth = ( uint ) Settings_Depth.Value;
            Settings.Default[ "Render_MaxDepth" ] = RayCaster.MaxDepth;
            Settings.Default.Save( );
        }

        private void Settings_Samples_ValueChanged( object sender, EventArgs e )
        {
            RayCaster.Samples = ( uint )Settings_Samples.Value;
            Settings.Default[ "Render_Samples" ] = RayCaster.Samples;
            Settings.Default.Save( );
        }

        private void Progress_Timer_Tick( object sender, EventArgs e )
        {
            if ( Renderer.Rendering )
            {
                Status_Progress.Value = Renderer.Done;
                float TicksInSeconds = ( Progress_Timer.Interval / 1000f );
                if ( AverageProgressPerSecond < 0 )
                    AverageProgressPerSecond = ( Status_Progress.Value - LastDone ) * TicksInSeconds;
                else
                {
                    AverageProgressPerSecond += ( Status_Progress.Value - LastDone ) * TicksInSeconds;
                    AverageProgressPerSecond =  AverageProgressPerSecond * ( 1f - TicksInSeconds );
                }
                LastDone = Status_Progress.Value;
                float Req = ( Renderer.Max - Status_Progress.Value );

                float TicksLeft = Req / AverageProgressPerSecond;
                float SecondsLeft = TicksLeft * TicksInSeconds;
                if ( !float.IsInfinity( SecondsLeft ) )
                    Status_Label.Text = Resources.Status_Rendering + ": " + TimeSpan.FromSeconds( ( int ) SecondsLeft );
            }
            else
            {
                Status_Label.Text = Resources.Status_Done;
                AverageProgressPerSecond = -1;
            }
        }
    }
}
