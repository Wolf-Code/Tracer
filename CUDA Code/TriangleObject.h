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
	return V1.Position;
}

#endif