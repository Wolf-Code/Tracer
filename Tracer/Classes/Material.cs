using System.ComponentModel;
using ManagedCuda.VectorTypes;
using Tracer.Classes.Util;
using Tracer.CUDA;

namespace Tracer.Classes
{
    [TypeConverter( typeof ( ExpandableObjectConverter ) )]
    [Description( "The object's material properties" )]
    public class Material
    {
        [Description( "The material's color." )]
        public Color Color { set; get; }

        [Description( "The material's radiance color and intensity." )]
        public Color Radiance { set; get; }

        [DisplayName( "Material type" )]
        [Description( "The material's surface type." )]
        public CUDAMaterialType Type { set; get; }

        [Description( "If the material is specular, this changes the glossyness of the reflection." )]
        public float Glossyness { set; get; }

        public Material( )
        {
            Radiance = Color.Black;
            Color = Color.White;
            Type = CUDAMaterialType.Diffuse;
            Glossyness = 0;
        }

        public CUDAMaterial ToCUDAMaterial( )
        {
            return new CUDAMaterial
            {
                Color = new float3( Color.R, Color.G, Color.B ),
                Radiance = new float3( Radiance.R, Radiance.G, Radiance.B ),
                Type = this.Type,
                Glossyness = this.Glossyness
            };
        }
    }
}