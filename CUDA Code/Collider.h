#pragma once
#ifndef __COLLIDER_H__
#define __COLLIDER_H__

#include "Defines.h"

class Collider
{
public:
    __device__ static CollisionResult Collide( Ray&,Object* );
    __device__ static CollisionResult SphereCollision(Ray&,Object* );
    __device__ static CollisionResult PlaneCollision(Ray&,Object* );
	__device__ static CollisionResult TriangleCollision(Ray&,Object* );
};

__device__ CollisionResult Collider::Collide( Ray& R, Object* Obj )
{
	CollisionResult Res;
	Res.Hit = false;

    switch ( Obj->Type )
    {
        case ObjectType::SphereType:
            Res = SphereCollision( R, Obj );
			break;

        case ObjectType::PlaneType:
            Res = PlaneCollision( R, Obj );
			break;

		case ObjectType::TriangleType:
			Res = TriangleCollision( R, Obj );
			break;
    }

	Res.Ray = &R;
	return Res;
}

__device__ CollisionResult Collider::SphereCollision( Ray& R, Object* Obj )
{
    CollisionResult Res;
    Res.Hit = false;

   SphereObject& Sphere = Obj->Sphere;

    const float A = VectorMath::Dot( R.Direction, R.Direction );
    const float B = 2 * VectorMath::Dot( R.Direction, R.Start - Sphere.Position );
    const float C = VectorMath::Dot( R.Start - Sphere.Position, R.Start - Sphere.Position ) - ( Sphere.Radius * Sphere.Radius );

    const float Discriminant = B * B - 4 * A * C;
    if ( Discriminant < 0 )
        return Res;

    const float DiscriminantSqrt = sqrt( Discriminant );
    float Q;
    if ( B < 0 )
        Q = ( -B - DiscriminantSqrt ) / 2.0;
    else
        Q = ( -B + DiscriminantSqrt ) / 2.0;

    float T0 = Q / A;
    float T1 = C / Q;

    if ( T0 > T1 )
    {
        float TempT0 = T0;
        T0 = T1;
        T1 = TempT0;
    }

    // Sphere is behind the ray's start position.
    if ( T1 < 0 )
        return Res;

    Res.Distance = T0 < 0 ? T1 : T0;
    Res.Hit = true;
    Res.Position = R.Start + R.Direction * Res.Distance;
    Res.Normal = VectorMath::Normalized( Res.Position - Sphere.Position );
    Res.HitObject = Obj;

    return Res;
}

__device__ CollisionResult Collider::PlaneCollision( Ray& R, Object* Obj )
{
	CollisionResult Res;
	Res.Hit = false;

	const PlaneObject& Plane = Obj->Plane;
	float Div = VectorMath::Dot( Plane.Normal, R.Direction );
	if ( Div == 0 )
		return Res;

	float Distance = -( VectorMath::Dot( Plane.Normal, R.Start ) + Plane.Offset ) / Div;
	if ( Distance < 0 )
		return Res;

	Res.Hit = true;
	Res.HitObject = Obj;
	Res.Distance = Distance;
	Res.Normal = Plane.Normal;
	Res.Position = R.Start + R.Direction * Distance;

	return Res;
}

__device__ CollisionResult Collider::TriangleCollision( Ray& R, Object* Obj )
{
	CollisionResult Res;
	Res.Hit = false;
	const TriangleObject& Triangle = Obj->Triangle;

	const float3 e1 = Triangle.V2.Position - Triangle.V1.Position;
	const float3 e2 = Triangle.V3.Position - Triangle.V1.Position;
	const float3 q = VectorMath::Cross( R.Direction, e2 );
	const float a = VectorMath::Dot( e1, q );
	//if(a < 0) return false; // Backface cull
	if ( a > -Epsilon && a < Epsilon ) return Res;

	const float f = 1.0f / a;
	const float3 s = R.Start - Triangle.V1.Position;
	const float u = f * VectorMath::Dot( s, q );
	if ( u < 0.0 || u > 1.0 ) return Res;

	const float3 _R = VectorMath::Cross( s, e1 );
	const float v = f * VectorMath::Dot( R.Direction, _R );
	if ( v < 0.0 || u + v > 1.0 ) return Res;

	const float t = f * VectorMath::Dot( e2, _R );
	if ( t < Epsilon ) return Res;

	Res.Distance = t;
	Res.Position = R.Start + R.Direction * t;
	const float w = 1.0f - ( u + v );
	const float3 Normal = Obj->Triangle.Normal;
	Res.Normal = VectorMath::Normalized( w*Normal + u*Normal + v*Normal );
	Res.Hit = true;
	Res.HitObject = Obj;

	return Res;
}

#endif