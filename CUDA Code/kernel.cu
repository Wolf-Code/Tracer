#include <device_launch_parameters.h>
#include "CUDAIncluder.h"
#include "Raytracer.h"

extern "C"
{
    __constant__ Object* ObjectArray;
    __constant__ unsigned int Objects;
	__constant__ unsigned int* Lights;
	__constant__ unsigned int LightCount;
    __constant__ CamData Camera;
    __constant__ long Seed;
	__constant__ unsigned int MaxDepth;

    __global__ void TraceKernelRegion( float3* Input, int StartX, int StartY, int Width, int Height, float3* Output )
    {
        //      Which block # of T in B      ID of Thread
        int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
        int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
		

        if ( x < Width && y < Height )
        {
			int ID = y * Width + x;

            curandState RandState;
            curand_init( Seed + ID, 0, 0, &RandState );
            
            float JitteredX = ( StartX + x ) + curand_uniform( &RandState );
            float JitteredY = ( StartY + y ) + curand_uniform( &RandState );

            Ray R = Camera.GetRay( JitteredX, JitteredY );
			R.Depth = 0;

			Raytracer Tracer = Raytracer( ObjectArray, Objects, Lights, LightCount, &RandState );
			Output[ ID ] = Input[ ID ] + Tracer.RadianceIterative( MaxDepth, R );
        }
    }
}

int main( )
{
    return 0;
}