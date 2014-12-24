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
        public Menu( )
        {
            InitializeComponent( );
        }

        protected override void OnLoad( EventArgs e )
        {
            base.OnLoad( e );
            Renderer.Initialize( this );

            Renderer.Cam.Position = new Vector3( 140, 30, 120 );
            Renderer.Cam.Angle = new Angle { Pitch = -10, Yaw = 0, Roll = 0f };

            for ( int Q = 0; Q < 5; Q++ )
            {
                float Sz = 20 + Q * 5;
                Sphere S = new Sphere( new Vector3( ( Sz + 20 ) * Q, Sz / 2f, -40 ), Sz )
                {
                    Material =
                    {
                        Radiance = Color.White,
                        Color = new Color(RNG.GetPositiveUnitFloat(  ),RNG.GetPositiveUnitFloat(  ),RNG.GetPositiveUnitFloat(  )),
                        Specular = 1f
                    }
                };
                RayCaster.Objects.Add( S );
            }

            // Floor
            RayCaster.Objects.Add( new Plane( new Vector3( 0, 1, 0 ), 0 )
            {
                Material = new Material
                {
                    Color = new Color( 1f, 1f, 1f ),
                    Specular = 20f
                }
            } );

            // Back
            RayCaster.Objects.Add( new Plane( new Vector3( 0, 0, 1 ), 70 )
            {
                Material = new Material
                {
                    Color = new Color( 1f, 1f, 1f ),
                    Specular = 20f
                }
            } );
            
            // Ceiling
            RayCaster.Objects.Add( new Plane( new Vector3( 0, -1, 0 ), 90 )
            {
                Material = new Material
                {
                    Color = new Color( 1f, 1f, 1f ),
                    Specular = 20f
                }
            } );

            // Left
            RayCaster.Objects.Add( new Plane( new Vector3( 1, 0, 0 ), 70 )
            {
                Material = new Material
                {
                    Color = new Color( 1f, 0f, 0f ),
                    Specular = 20f
                }
            } );

            // Right
            RayCaster.Objects.Add( new Plane( new Vector3( -1, 0, 0 ), 300 )
            {
                Material = new Material
                {
                    Color = new Color( 0f, 0f, 1f ),
                    Specular = 20f
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
                Filter = "PNG (*.png)|*.png",
                InitialDirectory = Settings.Default.Image_Folder,
                FileName = DateTime.Now.ToShortDateString( ) + "_" + DateTime.Now.ToLongTimeString(  ).Replace( ':', '-' ) + ".png"
            };

            if ( Dialog.ShowDialog( ) == DialogResult.OK )
                this.RenderImage.Image.Save( Dialog.FileName );
        }
    }
}
