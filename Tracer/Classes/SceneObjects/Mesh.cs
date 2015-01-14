using System;
using System.ComponentModel;
using System.Drawing.Design;
using System.Windows.Forms.Design;
using Tracer.Classes.Util;
using Tracer.Interfaces;
using Tracer.Structs.CUDA;
using Tracer.Utilities;

namespace Tracer.Classes.SceneObjects
{
    [Serializable]
    public class Mesh : GraphicsObject
    {
        [Editor( typeof ( FileNameEditor ), typeof ( UITypeEditor ) )]
        [Description( "The path to the model file" )]
        public string Path { set; get; }

        public Vector3 Position { set; get; }

        public Vector3 Scale { set; get; }

        public Mesh( )
        {
            Position = new Vector3( );
            Scale = new Vector3( 1, 1, 1 );
        }

        public override CUDAObject [ ] ToCUDA( )
        {
            IModel M = ModelImporter.Load( Path );
            M.SetPosition( Position );
            M.SetScale( Scale );

            CUDAObject [ ] Meshes = M.ToCuda( );

            return Meshes;
        }
    }
}