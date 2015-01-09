#pragma once
#ifndef __OBJECT_H__
#define __OBJECT_H__

#include "CUDAIncluder.h"
#include "ObjectType.h"
#include "SphereObject.h"
#include "PlaneObject.h"
#include "TriangleObject.h"
#include "Material.cuh"

struct Object
{
	unsigned int ID;
    ObjectType Type;
    SphereObject Sphere;
    PlaneObject Plane;
	TriangleObject Triangle;
    Material Material;
	__device__ bool IsLightSource( void );
};

__device__ bool operator==( Object& Obj1, Object& Obj2 );


__device__ bool Object::IsLightSource( void )
{
	return	this->Material.Radiance.x > 0 ||
			this->Material.Radiance.y > 0 ||
			this->Material.Radiance.z > 0;
}

__device__ bool operator==( Object& Obj1, Object& Obj2 )
{
	return Obj1.ID == Obj2.ID;
}

#endif