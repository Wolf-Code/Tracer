using System;
using System.ComponentModel;

namespace Tracer.Classes.Objects
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

        protected GraphicsObject( )
        {
            this.Material = new Material( );
        }

        public override string ToString( )
        {
            return this.Name;
        }
    }
}