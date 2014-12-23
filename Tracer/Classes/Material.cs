using Tracer.Classes.Util;

namespace Tracer.Classes
{
    public class Material
    {
        public Color Color;
        public Color Radiance;
        public float Shininess;

        public Material( )
        {
            Radiance = Color.Black;
            Color = Color.White;
            Shininess = 0f;
        }
    }
}
