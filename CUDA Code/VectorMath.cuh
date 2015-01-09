#pragma once
#ifndef __VECTORMATH_H__
#define __VECTORMATH_H__

#include "CUDAIncluder.h"

class VectorMath
{
public:
	__device__ static float Dot( const float3&, const float3& );
	__device__ static float3 Cross( const float3&, const float3& );
	__device__ static float Length( const float3& );
	__device__ static void Normalize( float3& );
	__device__ static float LargestComponent( const float3& );
	__device__ static float3 Normalized( const float3& );
	__device__ static float3 MakeVector( const float, const float, const float );
	__device__ static float3 MakeVector( const float );
	__device__ static float3 Reflect( const float3&, const float3& );
	__device__ static float3 RandomCosineDirectionInSameDirection( const float3&, curandState* );
	__device__ static float3 RandomDirectionInSameDirection( const float3&, curandState* );
	__device__ static float3 RandomDirection( curandState* );
};

__device__ float3 operator/( const float3&, const float );
__device__ float3 operator*( const float3&, const float );
__device__ float3 operator*( const float, const float3& );
__device__ float3 operator*( const float3&, const float3& );
__device__ float3 operator+( const float3&, const float3& );
__device__ void operator+=( float3&, const float3& );
__device__ void operator*=( float3&, const float3& );
__device__ float3 operator-( const float3&, const float3& );
__device__ bool operator==( const float3&, const float3& );


#endif