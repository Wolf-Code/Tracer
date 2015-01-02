using System;
using System.Windows.Forms;
using Tracer.Classes.Objects;
using Tracer.Classes.Util;
using Tracer.Properties;

namespace Tracer
{
    public partial class Menu : Form
    {
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

        private void ToolStrip_Button_Save_Click( object sender, EventArgs e )
        {
            if ( Renderer.Rendering ) return;
            if ( this.RenderImage.Image == null ) return;

            SaveFileDialog Dialog = new SaveFileDialog
            {
                Filter = "PNG (*.png)|*.png",
                FileName = DateTime.Now.ToShortDateString( ) + "_" + DateTime.Now.ToLongTimeString(  ).Replace( ':', '-' ) + ".png"
            };

            if ( Dialog.ShowDialog( ) == DialogResult.OK )
                this.RenderImage.Image.Save( Dialog.FileName );
        }

        private void Settings_Depth_ValueChanged( object sender, EventArgs e )
        {
            Settings.Default[ "Render_MaxDepth" ] = ( uint ) Settings_Depth.Value;
            Settings.Default.Save( );
        }

        private void Settings_Samples_ValueChanged( object sender, EventArgs e )
        {
            Settings.Default[ "Render_Samples" ] = ( uint )Settings_Samples.Value;
            Settings.Default.Save( );
        }

        private void Scene_SaveButton_Click( object sender, EventArgs e )
        {
            SaveFileDialog Dialog = new SaveFileDialog
            {
                Filter = "Path Tracer Scene (*.pts)|*.pts"
            };

            if ( Dialog.ShowDialog( ) != DialogResult.OK ) return;

            Content.Save( Dialog.FileName, Renderer.Scene );

            Settings.Default[ "Location_LastScene" ] = Dialog.FileName;
            Settings.Default.Save( );
        }

        private void Scene_LoadButton_Click( object sender, EventArgs e )
        {
            OpenFileDialog Dialog = new OpenFileDialog
            {
                Filter = "Path Tracer Scene (*.pts)|*.pts"
            };

            if ( Dialog.ShowDialog( ) != DialogResult.OK ) return;

            try
            {
                Renderer.Scene = Content.Load<Scene>( Dialog.FileName );
                SceneProperties.SelectedObject = Renderer.Scene;

                Settings.Default[ "Location_LastScene" ] = Dialog.FileName;
                Settings.Default.Save( );
            }
            catch
            {
                MessageBox.Show( "The loaded PTS file was not a valid path tracer scene." );
            }
        }


        private void Scene_LoadDefault_Click( object sender, EventArgs e )
        {
            Renderer.Scene = CUDAInterface.DefaultScene;
            SceneProperties.SelectedObject = Renderer.Scene;
        }
    }
}
