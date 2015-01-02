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
        }

        private void Button_Render_Click( object sender, EventArgs e )
        {
            if ( !Renderer.Rendering )
                Renderer.RenderImage( );
            else
                Renderer.CancelRendering( );
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
            Renderer.Scene = Scene.Default;
            SceneProperties.SelectedObject = Renderer.Scene;
        }
    }
}
