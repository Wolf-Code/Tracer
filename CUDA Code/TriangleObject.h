#pragma once
#ifndef __TRIANGLEOBJECT_H__
#define __TRIANGLEOBJECT_H__

#include "CUDAIncluder.h"
#include "Vertex.h"

struct TriangleObject
{
	Vertex V1, V2, V3;
	float3 Normal;
	__device__ float3 RandomPositionOnTriangle( curandState* );
};

__device__ float3 TriangleObject::RandomPositionOnTriangle( curandState* RandState ) 
{ 
	const float3 e1 = V2.Position - V1.Position;
	const float3 e2 = V3.Position - V1.Position;

	float R = curand_uniform( RandState );
	float S = curand_uniform( RandState );

	if ( R + S >= 1 )
	{
		R = 1 - R;
		S = 1 - S;
	}

	return V1.Position + e1 * R + e2 * S;
}

#endif