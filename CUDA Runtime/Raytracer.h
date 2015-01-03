#include "CUDAIncluder.h"
#include "VectorMath.h"
#include "MaterialType.h"
#include "Material.h"
#include "ObjectType.h"
#include "Ray.h"

#include "SphereObject.h"
#include "PlaneObject.h"
#include "Object.h"
#include "CollisionResult.h"
#include "Collider.h"
#include "CamData.h"

#define PI 3.1415926535
#define OneOverPI 0.31830988618
#define Bias 0.01

class Raytracer
{
public:
    __device__ static CollisionResult Trace( Ray*, Object*, int );
	__device__ static float CalculateBDRF( MaterialType, CollisionResult, Ray* );

    template <int>
    __device__ static float3 Radiance( Ray*, Object*, int, curandState* );
};

__device__ CollisionResult Raytracer::Trace( Ray* R, Object* Objects, int ObjectCount )
{ 
    CollisionResult Res = CollisionResult( );
    Res.Hit = false;

    for ( int Q = 0; Q < ObjectCount; Q++ )
    {
        CollisionResult TempRes = Collider::Collide( R, &Objects[ Q ] );
        if ( !Res.Hit || ( TempRes.Hit && TempRes.Distance < Res.Distance ) )
            Res = TempRes;
    }

    return Res;
}

__device__ float Raytracer::CalculateBDRF( MaterialType Type, CollisionResult Result, Ray* Ray )
{
	switch ( Type )
	{
		case Diffuse:
			float cos_theta = VectorMath::Dot( Ray->Direction, Result.Normal );
			return ( 2.0f * cos_theta ) * OneOverPI;
		case Reflective:
			return 1.0f;
		default:
			return 1.0f;
	}
}

template <int depth>
__device__ float3 Raytracer::Radiance( Ray* R, Object* Objects, int ObjectCount, curandState* RandState )
{
    const CollisionResult Res = Raytracer::Trace( R, Objects, ObjectCount );
    if ( !Res.Hit )
        return float3( );

	const Material Mat = Res.HitObject->Material;

    const float3 Rad = Mat.Radiance;
    if ( Rad.x >= 1 || Rad.y >= 1 || Rad.z >= 1 )
        return Rad;

    R->Depth += 1;
    R->Start = Res.Position + Res.Normal * Bias;

	float3 RandomDirection = VectorMath::RandomDirectionInSameDirection( Res.Normal, RandState );

	switch ( Mat.Type )
	{
		case Diffuse:
			R->Direction = RandomDirection;
			break;

		case Reflective:
			float3 Ref = VectorMath::Reflect( R->Direction, Res.Normal );
			float Glossyness = Mat.Glossyness;
			if ( Glossyness > 0 )
			{
				float3 Rand = RandomDirection;
				Ref = VectorMath::Normalized( Ref + Rand * Glossyness );
			}
			R->Direction = Ref;
			break;
	}

	return Rad + Mat.Color * ( Radiance<depth + 1>( R, Objects, ObjectCount, RandState ) * CalculateBDRF( Mat.Type, Res, R ) );
}

template<>
__device__ float3 Raytracer::Radiance<5>( Ray* R, Object* Objects, int ObjectCount, curandState* RandState )
{
    return float3( );
}