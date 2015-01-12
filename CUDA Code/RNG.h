#pragma once
#ifndef __RNG_H__
#define __RNG_H__

#include "CUDAIncluder.h"

class RNG
{
public:
	__device__ static void Initialize( curandState* RandState );
	__device__ static float Uniform( void );
	__device__ static float Next( float Max );
	__device__ static unsigned int NextUnsignedInt( unsigned int Max );

private:
	curandState* RandState;
};

__device__ void RNG::Initialize( curandState* RandState )
{
	RNG::RandState = RandState;
}

__device__ float RNG::Uniform( )
{
	return curand_uniform( RandState );
}

__device__ float RNG::Next( float Max )
{
	return RNG::Uniform( ) * Max;
}

__device__ unsigned int RNG::NextUnsignedInt( unsigned int Max )
{
	return ( unsigned int )RNG::Next( Max );
}

#endif