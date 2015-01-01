#include "CUDAIncluder.h"
#include "Raytracer.h"

extern "C"
{
    __device__ float clamp( float, float, float );

    __constant__ Object* ObjectArray;
    __constant__ unsigned int Objects;
    __constant__ CamData Camera;
    __constant__ long Seed;
    /*
    __global__ void TraceKernel( float3* Input, float3* Output )
    {
        //      Which block # of T in B      ID of Thread
        int x = ( blockIdx.x * blockDim.x ) + threadIdx.x;
        int y = ( blockIdx.y * blockDim.y ) + threadIdx.y;

        if ( x < Camera.Width && y < Camera.Height )
        {
            int ID = y * ( int )Camera.Width + x;

            curandState RandState;
            curand_init( Seed + ID, 0, 0, &RandState );

            Ray R = Camera.GetRay( x, y );

            Output[ ID ] = Input[ ID ] + Radiance<0>( &R, ObjectArray, Objects, &RandState );
        }
    }
    */

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
            
            float JitteredX = clamp( x + ( curand_uniform( &RandState ) * 2.0f - 1.0f ) * 0.5f, 0, Camera.Width );
            float JitteredY = clamp( y + ( curand_uniform( &RandState ) * 2.0f - 1.0f ) * 0.5f, 0, Camera.Height );

            Ray R = Camera.GetRay( JitteredX, JitteredY );

            Output[ ID ] = Input[ ID ] + Radiance<0>( &R, ObjectArray, Objects, &RandState );
        }
    }
}

int main( )
{
    return 0;
}