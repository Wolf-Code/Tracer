#pragma once
#ifndef __SPHEREOBJECT_H__
#define __SPHEREOBJECT_H__

struct SphereObject
{
public:
    float3 Position;
    float Radius;
	__device__ float3 RandomPositionOnSphere( curandState* );
};

__device__ float3 SphereObject::RandomPositionOnSphere( curandState* RandState )
{
	return Position + VectorMath::RandomDirection( RandState ) * Radius;
}

#endif