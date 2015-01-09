#pragma once
#ifndef __COLLISIONRESULT_H__
#define __COLLISIONRESULT_H__

struct CollisionResult
{
    bool Hit;
    float3 Position;
    float3 Normal;
    float Distance;
    Object* HitObject;
	Ray* Ray;
};

#endif