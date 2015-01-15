#pragma once
#ifndef __MESHOBJECT_H__
#define __MESHOBJECT_H__
#include "TriangleObject.h"

struct MeshObject
{
	TriangleObject* Triangles;
	SphereObject BoundingVolume;
	unsigned int TriangleCount;
	__device__ float3 RandomPositionOnMesh( curandState* );
};

__device__ float3 MeshObject::RandomPositionOnMesh( curandState* RandState )
{
	unsigned int ID = unsigned int( roundf( curand_uniform( RandState ) * ( TriangleCount - 1 ) ) );

	TriangleObject& T = Triangles[ ID ];
	return T.RandomPositionOnTriangle( RandState );
}

#endif