#include "CUDAIncluder.h"
#include "curand.h"
#include "curand_kernel.h"
#include "VectorMath.h"
#include "Material.h"
#include "TypeClass.h"
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
    __device__ static float3 TraceColor( Ray R, Object* Objects, int ObjectCount );
    __device__ static float3 Radiance( Ray R, Object* Objects, int ObjectCount, curandState* RandState );
    __device__ static float3 RadianceIterative( Ray R, Object* Objects, int ObjectCount, curandState* RandState );

private:
    __device__ static CollisionResult Trace( Ray R, Object* Objects, int ObjectCount );
    __device__ static float3 RandomCosineDirectionInSameDirection( float3 Direction, curandState* RandState );
};

__device__ float3 Raytracer::RandomCosineDirectionInSameDirection( float3 Direction, curandState* RandState )
{
    float3 Rand = float3( );

    Rand.x = curand_uniform( RandState ) * 2.0 - 1.0;
    Rand.y = curand_uniform( RandState ) * 2.0 - 1.0;
    Rand.z = curand_uniform( RandState ) * 2.0 - 1.0;
    Rand = VectorMath::Normalized( Rand );

    if ( VectorMath::Dot( Direction, Rand ) < 0 )
        Rand = Rand * -1;
    
    Rand = VectorMath::Normalized( Rand + Direction );

    return Rand;
}

__device__ CollisionResult Raytracer::Trace( Ray R, Object* Objects, int ObjectCount )
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

__device__ float3 Raytracer::RadianceIterative( Ray R, Object* Objects, int ObjectCount, curandState* RandState )
{
    const int MaxDepth = 3;
    float3 PreviousRads[ MaxDepth ];
    PreviousRads[ 0 ] = float3( );
    float3 PreviousMatColors[ MaxDepth ];
    PreviousMatColors[ 0 ] = float3( );
    float  PreviousBDRFs[ MaxDepth ];
    PreviousBDRFs[ 0 ] = 0;

    while ( R.Depth < MaxDepth )
    {
        CollisionResult Res = Raytracer::Trace( R, Objects, ObjectCount );
        if ( !Res.Hit )
            break;

        float3 Rad = Res.HitObject.Material.Radiance;
        PreviousRads[ R.Depth ] = Rad;
        if ( Rad.x >= 1 || Rad.y >= 1 || Rad.z >= 1 )
        {
            PreviousMatColors[ R.Depth ] = float3( );
            PreviousBDRFs[ R.Depth ] = 0;
            R.Depth = R.Depth + 1;
            break;
        }

        Ray R2 = Ray( );
        R2.Depth = R.Depth + 1;
        R2.Direction = Raytracer::RandomCosineDirectionInSameDirection( Res.Normal, RandState );
        R2.Start = Res.Position + Res.Normal;

        float cos_theta = VectorMath::Dot( R2.Direction, Res.Normal );
        if ( cos_theta < 0 )
            cos_theta = 0;

        float BDRF = 2 * cos_theta;

        PreviousMatColors[ R.Depth ] = Res.HitObject.Material.Color;
        PreviousBDRFs[ R.Depth ] = BDRF;

        if ( R2.Depth >= MaxDepth )
            break;

        R = R2;
    }

    float3 Last = PreviousRads[ R.Depth - 1 ];
    for ( int Q = R.Depth - 2; Q >= 0; Q-- )
    {
        Last = PreviousRads[ Q ] + PreviousMatColors[ Q ] * ( Last * PreviousBDRFs[ Q ] );
    }

    return Last;
}

__device__ float3 Raytracer::Radiance( Ray R, Object* Objects, int ObjectCount, curandState* RandState )
{
    if ( R.Depth > 5 )
        return float3( );

    CollisionResult Res = Raytracer::Trace( R, Objects, ObjectCount );
    if ( !Res.Hit )
        return float3( );

    float3 Rad = Res.HitObject.Material.Radiance;
    if ( Rad.x >= 1 || Rad.y >= 1 || Rad.z >= 1 )
        return Rad;

    Ray R2 = Ray( );
    R2.Depth = R.Depth + 1;
    R2.Direction = Raytracer::RandomCosineDirectionInSameDirection( Res.Normal, RandState );
    R2.Start = Res.Position + Res.Normal;

    float cos_theta = VectorMath::Dot( R.Direction, Res.Normal );
    if ( cos_theta < 0 )
        cos_theta = 0;

    float BDRF = 2 * cos_theta;

    return Rad + Res.HitObject.Material.Color * ( Raytracer::Radiance( R2, Objects, ObjectCount, RandState ) * BDRF );
}

__device__ float3 Raytracer::TraceColor( Ray R, Object* Objects, int ObjectCount )
{
    CollisionResult Res = Trace( R, Objects, ObjectCount );

    if ( !Res.Hit )
        return float3( );
    else
        return Res.HitObject.Material.Color;
}