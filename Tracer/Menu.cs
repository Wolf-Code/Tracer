using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
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
        public Menu( )
        {
            InitializeComponent( );
        }

        protected override void OnLoad( EventArgs e )
        {
            base.OnLoad( e );
            Renderer.Initialize( this );

            RayCaster.Objects.Add( new Sphere( Renderer.Cam.Angle.Forward * 70, 20 ) );
            RayCaster.Objects.Add( new Sphere( Renderer.Cam.Angle.Forward * 40 + Renderer.Cam.Angle.Up * 5 + Renderer.Cam.Angle.Right * 15, 10 )
            {
                Material = new Material { Color = new Color( 255, 0, 0 ) }
            } );
            RayCaster.Objects.Add( new Sphere( Renderer.Cam.Angle.Forward * 42 + Renderer.Cam.Angle.Up * 25, 3f )
            {
                Material = new Material { Color = new Color( 0, 255, 0 ) }
            } );
            RayCaster.Objects.Add( new Plane( new Vector3( 0, 1, 0 ), 20 ) );

            RayCaster.Lights.Add( new Light
            {
                DiffuseColor = new Color( 255, 255, 255 ),
                Position = Renderer.Cam.Angle.Forward * 48 + Renderer.Cam.Angle.Up * 38,
                FallOffDistance = 150,
                AmbientIntensity = .4f,
                Intensity = 1f
            } );
        }

        private void Button_Render_Click( object sender, EventArgs e )
        {
            Renderer.RenderImage( );
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
                Filter = "*PNG (*.png)|*.png",
                InitialDirectory = Settings.Default.Image_Folder,
                FileName = DateTime.Now.ToShortDateString( ) + "_" + DateTime.Now.ToLongTimeString(  ).Replace( ':', '-' ) + ".png"
            };
            if ( Dialog.ShowDialog( ) == DialogResult.OK )
            {
                this.RenderImage.Image.Save( Dialog.FileName );
            }
        }
    }
}
