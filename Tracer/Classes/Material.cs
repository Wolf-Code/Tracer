using Tracer.Classes.Util;

namespace Tracer.Classes
{
    public class Material
    {
        public Color Color;
        public float Shininess;

        public Material( )
        {
            Color = new Color( 1f, 1f, 1f );
            Shininess = 0f;
        }
    }
}
