using System;
using System.ComponentModel;
using Tracer.Classes.Util;
using Tracer.Interfaces;
using Tracer.Structs.CUDA;
using Tracer.Utilities;

namespace Tracer.Classes.SceneObjects
{
    [Serializable]
    public class Mesh : GraphicsObject
    {
        [Editor( typeof ( System.Windows.Forms.Design.FileNameEditor ), typeof ( System.Drawing.Design.UITypeEditor ) )]
        [Description( "The path to the model file" )]
        public string Path { set; get; }

        public Vector3 Position { set; get; }

        public Vector3 Scale { set; get; }

        public Mesh( )
        {
            this.Position = new Vector3( );
            this.Scale = new Vector3( 1, 1, 1 );
        }

        public override CUDAObject [ ] ToCUDA( )
        {
            IModel M = ModelImporter.Load( Path );
            M.SetPosition( this.Position );
            M.SetScale( this.Scale );

            CUDAObject [ ] Meshes = M.ToCuda( );

            return Meshes;
        }
    }
}