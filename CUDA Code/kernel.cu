#include <device_launch_parameters.h>
#include "CUDAIncluder.h"
#include "Raytracer.h"

extern "C"
{
    __device__ __constant__ Object* ObjectArray;
    __device__ __constant__ unsigned int Objects;
	__device__ __constant__ unsigned int* Lights;
	__device__ __constant__ unsigned int LightCount;
	__device__ __constant__ CamData Camera;
	__device__ __constant__ unsigned int MaxDepth;

    __global__ void TraceKernelRegion( unsigned int Samples, long Seed, float3* Input, int StartX, int StartY, int Width, int Height, float3* Output )
    {
        //      Which block # of T in B      ID of Thread
        int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
        int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;
		

        if ( x < Width && y < Height )
        {
			int ID = y * Width + x;

            curandState RandState;
            curand_init( Seed + ID, 0, 0, &RandState );
			float3 Val = Input[ ID ];

			Raytracer Tracer = Raytracer( ObjectArray, Objects, Lights, LightCount, &RandState );
			for ( int Q = 0; Q < Samples; Q++ )
			{
				float JitteredX = ( StartX + x ) + curand_uniform( &RandState );
				float JitteredY = ( StartY + y ) + curand_uniform( &RandState );

				Ray R = Camera.GetRay( JitteredX, JitteredY );
				R.Depth = 0;

				Val += Tracer.RadianceIterative( MaxDepth, R );
			}

			Output[ ID ] = Val;
        }
    }
}

int main( )
{
    return 0;
}