#include "CUDAIncluder.h"
#include "curand.h"
#include "curand_kernel.h"
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
    __device__ static float3 TraceColor( Ray*, Object*, int );
    //__device__ static float3 Radiance( Ray R, Object* Objects, int ObjectCount, curandState* RandState );
    __device__ static float3 RadianceIterative( Ray*, Object*, int, curandState* );
    __device__ static CollisionResult Trace( Ray*, Object*, int );
    __device__ static float3 RandomCosineDirectionInSameDirection( float3, curandState* );
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
/*
__device__ float3 Raytracer::RadianceIterative( Ray* R, Object* Objects, int ObjectCount, curandState* RandState )
{
    const int MaxDepth = 3;
    float3 PreviousRads[ MaxDepth ];
    PreviousRads[ 0 ] = float3( );
    float3 PreviousMatColors[ MaxDepth ];
    PreviousMatColors[ 0 ] = float3( );
    float  PreviousBDRFs[ MaxDepth ];
    PreviousBDRFs[ 0 ] = 0;

    while ( R->Depth < MaxDepth )
    {
        CollisionResult Res = Raytracer::Trace( R, Objects, ObjectCount );
        if ( !Res.Hit )
            break;

        float3 Rad = Res.HitObject.Material.Radiance;
        PreviousRads[ R->Depth ] = Rad;
        if ( Rad.x >= 1 || Rad.y >= 1 || Rad.z >= 1 )
        {
            PreviousMatColors[ R->Depth ] = float3( );
            PreviousBDRFs[ R->Depth ] = 0;
            R->Depth = R->Depth + 1;
            break;
        }

        Ray R2 = Ray( );
        R2.Depth = R->Depth + 1;
        R2.Direction = Raytracer::RandomCosineDirectionInSameDirection( Res.Normal, RandState );
        R2.Start = Res.Position + Res.Normal;

        float cos_theta = VectorMath::Dot( R2.Direction, Res.Normal );
        float BDRF = 2 * cos_theta;

        PreviousMatColors[ R->Depth ] = Res.HitObject.Material.Color;
        PreviousBDRFs[ R->Depth ] = BDRF;

        if ( R2.Depth >= MaxDepth )
            break;

        R = &R2;
    }

    float3 Last = PreviousRads[ R->Depth - 1 ];
    for ( int Q = R->Depth - 2; Q >= 0; Q-- )
    {
        Last = PreviousRads[ Q ] + PreviousMatColors[ Q ] * ( Last * PreviousBDRFs[ Q ] );
    }

    return Last;
}
*/

template <int depth>
__device__ float3 Radiance( Ray* R, Object* Objects, int ObjectCount, curandState* RandState )
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
__device__ float3 Radiance<5>( Ray* R, Object* Objects, int ObjectCount, curandState* RandState )
{
    return float3( );
}