#pragma once
#ifndef __MESHOBJECT_H__
#define __MESHOBJECT_H__
#include "TriangleObject.h"

struct MeshObject
{
	TriangleObject* Triangles;
	SphereObject BoundingVolume;
	unsigned int TriangleCount;
};

#endif