using Tracer.Classes.Util;

namespace Tracer.Classes.Objects
{
    class LightSphere : Sphere
    {
        public LightSphere( Vector3 Position, float Radius ) : base( Position, Radius )
        {
            this.Material.Radiance = new Color( 1f, 1f, 1f );
        }
    }
}
