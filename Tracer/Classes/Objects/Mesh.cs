using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using Tracer.CUDA;
using Tracer.Importers;
using Tracer.Interfaces;

namespace Tracer.Classes.Objects
{
    [Serializable]
    public class Mesh : GraphicsObject
    {
        [Editor( typeof ( System.Windows.Forms.Design.FileNameEditor ), typeof ( System.Drawing.Design.UITypeEditor ) )]
        [Description( "The path to the model file" )]
        public string Path { set; get; }

        public override CUDAObject [ ] ToCUDA( )
        {
            IModel M = ModelImporter.Load( Path );
            Triangle [ ] Triangles = M.ToTriangles( );

            List<CUDAObject> Objs = new List<CUDAObject>( );
            foreach ( CUDAObject [ ] TObjs in Triangles.Select( T => T.ToCUDA( ) ) )
            {
                Objs.AddRange( TObjs );
            }

            return Objs.ToArray( );
        }
    }
}