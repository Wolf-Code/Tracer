#pragma once
#ifndef __COLLIDER_H__
#define __COLLIDER_H__

#include "Defines.h"

class Collider
{
public:
	__device__ static CollisionResult Collide( Ray&, Object* );
	__device__ static CollisionResult SphereCollision( Ray&, const SphereObject& );
	__device__ static CollisionResult PlaneCollision( Ray&, const PlaneObject& );
	__device__ static CollisionResult TriangleCollision( Ray&, const Vertex&, const Vertex&, const Vertex& );
	__device__ static CollisionResult TriangleCollision( Ray&, const TriangleObject& );
	__device__ static CollisionResult MeshCollision( Ray&, const MeshObject& );
};

__device__ CollisionResult Collider::Collide( Ray& R, Object* Obj )
{
	CollisionResult Res;
	Res.Hit = false;

    switch ( Obj->Type )
    {
        case ObjectType::SphereType:
            Res = SphereCollision( R, Obj->Sphere );
			break;

        case ObjectType::PlaneType:
            Res = PlaneCollision( R, Obj->Plane );
			break;

		case ObjectType::TriangleType:
			Res = TriangleCollision( R, Obj->Triangle );
			break;

		case ObjectType::MeshType:
			if ( SphereCollision( R, Obj->Mesh.BoundingVolume ).Hit )
				Res = MeshCollision( R, Obj->Mesh );
			break;
    }

	Res.Ray = &R;
	Res.HitObject = Obj;
	return Res;
}

__device__ CollisionResult Collider::MeshCollision( Ray& R, const MeshObject& Mesh )
{
	CollisionResult Res;
	Res.Hit = false;

	for ( int Q = 0; Q < Mesh.TriangleCount; Q++ )
	{
		const TriangleObject& T = Mesh.Triangles[ Q ];
		CollisionResult TempRes = Collider::TriangleCollision( R, T.V1, T.V2, T.V3 );
		if ( !Res.Hit || ( TempRes.Hit && TempRes.Distance < Res.Distance ) )
			Res = TempRes;
	}

	return Res;
}

__device__ CollisionResult Collider::SphereCollision( Ray& R, const SphereObject& Sphere )
{
    CollisionResult Res;
    Res.Hit = false;

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

    return Res;
}

__device__ CollisionResult Collider::PlaneCollision( Ray& R, const PlaneObject& Plane )
{
	CollisionResult Res;
	Res.Hit = false;

	float Div = VectorMath::Dot( Plane.Normal, R.Direction );
	if ( Div == 0 )
		return Res;

	float Distance = -( VectorMath::Dot( Plane.Normal, R.Start ) + Plane.Offset ) / Div;
	if ( Distance < 0 )
		return Res;

	Res.Hit = true;
	Res.Distance = Distance;
	Res.Normal = Plane.Normal;
	Res.Position = R.Start + R.Direction * Distance;

	return Res;
}

__device__ CollisionResult Collider::TriangleCollision( Ray& R, const Vertex& V1, const Vertex& V2, const Vertex& V3 )
{
	CollisionResult Res;
	Res.Hit = false;

	const float3 e1 = V2.Position - V1.Position;
	const float3 e2 = V3.Position - V1.Position;
	const float3 q = VectorMath::Cross( R.Direction, e2 );
	const float a = VectorMath::Dot( e1, q );
	//if(a < 0) return false; // Backface cull
	if ( a > -Epsilon && a < Epsilon ) return Res;

	const float f = 1.0f / a;
	const float3 s = R.Start - V1.Position;
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
	Res.Normal = VectorMath::Normalized( w*V1.Normal + u*V2.Normal + v*V3.Normal );

	if ( VectorMath::Dot( R.Direction, Res.Normal ) > 0 )
		Res.Normal = Res.Normal * -1;

	Res.Hit = true;

	return Res;
}

__device__ CollisionResult Collider::TriangleCollision( Ray& R, const TriangleObject& Triangle )
{
	CollisionResult Res = TriangleCollision( R, Triangle.V1, Triangle.V2, Triangle.V3 );

	return Res;
}

#endif