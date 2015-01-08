#include "Raytracer2.h"

extern "C"
{
    __device__ float clamp( float, float, float );

    __constant__ Object* ObjectArray;
    __constant__ unsigned int Objects;
	__constant__ Object* Lights;
	__constant__ unsigned int LightCount;
    __constant__ CamData Camera;
    __constant__ long Seed;
	__constant__ unsigned int MaxDepth;

    __device__ float clamp( float X, float Min, float Max )
    {
        return fmaxf( Min, fminf( X, Max ) );
    }

    __global__ void TraceKernelRegion( float3* Input, int StartX, int StartY, int EndX, int EndY, float3* Output )
    {
        //      Which block # of T in B      ID of Thread
        int x = StartX + ( blockIdx.x * blockDim.x ) + threadIdx.x;
        int y = StartY + ( blockIdx.y * blockDim.y ) + threadIdx.y;

        if ( x < EndX && y < EndY )
        {
            int ID = y * ( int )Camera.Width + x;

            curandState RandState;
            curand_init( Seed + ID, 0, 0, &RandState );
            
            float JitteredX = x + ( curand_uniform( &RandState ) * 2.0f - 1.0f ) * 0.5f;
            float JitteredY = y + ( curand_uniform( &RandState ) * 2.0f - 1.0f ) * 0.5f;

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