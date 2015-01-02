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

class Raytracer
{
public:
    __device__ static CollisionResult Trace( Ray*, Object*, int );

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

template <int depth>
__device__ float3 Raytracer::Radiance( Ray* R, Object* Objects, int ObjectCount, curandState* RandState )
{
    CollisionResult Res = Raytracer::Trace( R, Objects, ObjectCount );
    if ( !Res.Hit )
        return float3( );

    float3 Rad = Res.HitObject->Material.Radiance;
    if ( Rad.x >= 1 || Rad.y >= 1 || Rad.z >= 1 )
        return Rad;

    R->Depth += 1;
    R->Start = Res.Position + Res.Normal;

	float3 RandomDirection = VectorMath::RandomCosineDirectionInSameDirection( Res.Normal, RandState );

	float BDRF = 1.0f;

	switch ( Res.HitObject->Material.Type )
	{
		case MaterialType::Diffuse:
			R->Direction = RandomDirection;

			float cos_theta = VectorMath::Dot( R->Direction, Res.Normal );
			BDRF = BDRF = ( 2.0f * cos_theta ) / PI;
			break;

		case MaterialType::Reflective:
			float3 Ref = VectorMath::Reflect( R->Direction, Res.Normal );
			float Glossyness = Res.HitObject->Material.Glossyness;
			if ( Glossyness > 0 )
			{
				float3 Rand = RandomDirection;
				Ref = VectorMath::Normalized( Ref + Rand * Glossyness );
			}
			R->Direction = Ref;
			break;
	}

    return Rad + Res.HitObject->Material.Color * ( Radiance<depth + 1>( R, Objects, ObjectCount, RandState ) * BDRF );
}

template<>
__device__ float3 Raytracer::Radiance<5>( Ray* R, Object* Objects, int ObjectCount, curandState* RandState )
{
    return float3( );
}