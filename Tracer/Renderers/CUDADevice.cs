using ManagedCuda;

namespace Tracer.Renderers
{
    public class CUDADevice : IDevice
    {
        public CudaDeviceProperties Device;

        public string Name
        {
            get { return Device.DeviceName; }
        }

        public override string ToString( )
        {
            return Name;
        }
    }
}