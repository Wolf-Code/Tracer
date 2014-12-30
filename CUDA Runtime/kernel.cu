#include "CUDAIncluder.h"
#include "Raytracer.h"

extern "C"
{
    __constant__ Object* ObjectArray;
    __constant__ unsigned int Objects;
    __constant__ CamData Camera;
    __constant__ int Seed;

    __global__ void TraceKernel( float3* Output )
    {
        //      Which block  # of T in B  ID of Thread
        int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
        int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
        int ID = y * ( int )Camera.Width + x;

        if ( ID < Camera.Width * Camera.Height )
        {
            curandState RandState;
            curand_init( Seed + ID, 0, 0, &RandState );

            Ray R = Camera.GetRay( x, y );

            Output[ ID ] = Raytracer::RadianceIterative( R, ObjectArray, Objects, &RandState );
        }
    }
}

int main( )
{
    return 0;
}