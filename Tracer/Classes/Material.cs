using ManagedCuda.VectorTypes;
using Tracer.Classes.Util;
using Tracer.CUDA;

namespace Tracer.Classes
{
    public class Material
    {
        public Color Color;
        public Color Radiance;

        public Material( )
        {
            Radiance = Color.Black;
            Color = Color.White;
        }

        public CUDAMaterial ToCUDAMaterial( )
        {
            return new CUDAMaterial
            {
                Color = new float3( Color.R, Color.G, Color.B ),
                Radiance = new float3( Radiance.R, Radiance.G, Radiance.B )
            };
        }
    }
}
