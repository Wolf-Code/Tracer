using System;
using System.ComponentModel;
using Tracer.Structs.CUDA;

namespace Tracer.Classes.SceneObjects
{
    /// <summary>
    /// An abstract class for all objects which we can represent in the raytraced image.
    /// </summary>
    [Serializable]
    public abstract class GraphicsObject
    {
        /// <summary>
        /// The material of the object.
        /// </summary>
        [Category( "Appearance" )]
        public Material Material { set; get; }

        /// <summary>
        /// The name to identify the object with.
        /// </summary>
        [Category( "Identification" )]
        public string Name { set; get; }

        [Description( "Whether the item should be used during rendering." )]
        public bool Enabled { set; get; }

        protected GraphicsObject( )
        {
            this.Enabled = true;
            this.Material = new Material( );
        }

        public abstract CUDAObject [ ] ToCUDA( );

        public override string ToString( )
        {
            return this.Name;
        }
    }
}