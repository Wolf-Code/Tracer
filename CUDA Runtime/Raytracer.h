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

class Raytracer
{
public:
    __device__ static CollisionResult Trace( Ray*, Object*, int );
    __device__ static float3 RandomCosineDirectionInSameDirection( float3, curandState* );

    template <int>
    __device__ static float3 Radiance( Ray*, Object*, int, curandState* );
};

__device__ float3 Raytracer::RandomCosineDirectionInSameDirection( float3 Direction, curandState* RandState )
{
    float3 Rand = VectorMath::MakeVector( curand_uniform( RandState ) * 2.0f - 1.0f,
                                          curand_uniform( RandState ) * 2.0f - 1.0f,
                                          curand_uniform( RandState ) * 2.0f - 1.0f );

    Rand = VectorMath::Normalized( Rand );

    if ( VectorMath::Dot( Direction, Rand ) < 0 )
        Rand = Rand * -1;
    
    Rand = VectorMath::Normalized( Rand + Direction );

    return Rand;
}

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

    switch ( Res.HitObject->Material.Type )
    {
        case MaterialType::Diffuse:
            R->Direction = Raytracer::RandomCosineDirectionInSameDirection( Res.Normal, RandState );
            break;

        case MaterialType::Reflective:
            float3 Ref = VectorMath::Reflect( R->Direction, Res.Normal );
            float Glossyness = Res.HitObject->Material.Glossyness;
            if ( Glossyness > 0 )
            {
                float3 Rand = Raytracer::RandomCosineDirectionInSameDirection( Res.Normal, RandState );
                Ref = VectorMath::Normalized( Ref + Rand * Glossyness );
            }
            R->Direction = Ref;
            break;
    }

    float cos_theta = VectorMath::Dot( R->Direction, Res.Normal );
    float BDRF = 2.0f * cos_theta;

    return Rad + Res.HitObject->Material.Color * ( Radiance<depth + 1>( R, Objects, ObjectCount, RandState ) * BDRF );
}

template<>
__device__ float3 Raytracer::Radiance<5>( Ray* R, Object* Objects, int ObjectCount, curandState* RandState )
{
    return float3( );
}