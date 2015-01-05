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
	__device__ Raytracer( Object*, unsigned int, Object*, unsigned int, curandState* );
    __device__ CollisionResult Trace( Ray* );
	__device__ float CalculateBDRF( MaterialType, CollisionResult, Ray* );
	__device__ Object* GetRandomLight( );
	__device__ float3 ShadowRay( CollisionResult* );
	__device__ float3 RadianceIterative( unsigned int, Ray* );
	__device__ float3 TraceDirectLight( Ray* );
	__device__ float3 LEnvironment( Ray* );

    template <int>
    __device__ float3 Radiance( Ray* );
private:
	curandState* RandState;
	Object* Objects;
	unsigned int ObjectCount;
	Object* Lights;
	unsigned int LightCount;
};

__device__ Raytracer::Raytracer( Object* Objects, unsigned int ObjectCount, Object* Lights, unsigned int LightCount, curandState* RandState )
{
	this->Objects = Objects;
	this->ObjectCount = ObjectCount;
	this->Lights = Lights;
	this->LightCount = LightCount;
	this->RandState = RandState;
}

__device__ CollisionResult Raytracer::Trace( Ray* R )
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

__device__ float3 Raytracer::LEnvironment( Ray* Ray )
{
	return float3( );
}

__device__ float3 Raytracer::TraceDirectLight( Ray* Ray )
{
	CollisionResult Res = this->Trace( Ray );
	float3 L = float3( );
	float3 Throughput = VectorMath::MakeVector( 1, 1, 1 );

	while ( Ray->Depth < 5 )
	{
		if ( !Res.Hit )
		{
			L += LEnvironment( Ray );
			break;
		}

		L += ShadowRay( &Res ) * Res.HitObject->Material.Color;
	}

	return L;
}

__device__ float Raytracer::CalculateBDRF( MaterialType Type, CollisionResult Result, Ray* Ray )
{
	switch ( Type )
	{
		case Diffuse:
			{
				float cos_theta = VectorMath::Dot( Ray->Direction, Result.Normal );
				return ( 2.0f * cos_theta ) * OneOverPI;
			}
		case Reflective:
			return 1.0f;
		default:
			return 1.0f;
	}
}

__device__ Object* Raytracer::GetRandomLight( )
{
	unsigned int ID = ( unsigned int )roundf( curand_uniform( RandState ) * ( LightCount - 1 ) );
	return &Objects[ 0 ];
}

__device__ float3 Raytracer::ShadowRay( CollisionResult* Result )
{
	Object* L = Raytracer::GetRandomLight( );

	float3 Start = Result->Position + Result->Normal * Bias;

	Ray R;
	R.Start = Start;

	switch ( L->Type )
	{
		case SphereType:
			{
				float3 RandomPos = L->Sphere.RandomPositionOnSphere( RandState );
				R.Direction = VectorMath::Normalized( RandomPos - Start ); 
			}
			break;

		case PlaneType:
			R.Direction = L->Plane.Normal * -1;
			break;

		default:
			return float3( );
	}

	const CollisionResult Res = Raytracer::Trace( &R );

	if ( Res.HitObject->ID == L->ID )
		return L->Material.Radiance * abs( VectorMath::Dot( R.Direction, Result->Normal ) );

	return float3( );
}

__device__ float3 Raytracer::RadianceIterative( unsigned int MaxDepth, Ray* R )
{
	int Depth = 0;
	float3 Val = float3( );
	float3 Throughput = VectorMath::MakeVector( 1, 1, 1 );

	while ( Depth < MaxDepth )
	{
		CollisionResult Res = this->Trace( R );
		if ( !Res.Hit )
			return Val;

		const Material Mat = Res.HitObject->Material;

		if ( Res.HitObject->IsLightSource( ) && Depth == 0 )
			return Mat.Radiance;

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
				R->Depth = 0;
				break;
		}
		Depth++;

		if ( Res.HitObject->Material.Type == Reflective )
			continue;

		Val += Raytracer::ShadowRay( &Res ) * Res.HitObject->Material.Color * Throughput;
	}

	return Val;
}

template <int depth>
__device__ float3 Raytracer::Radiance( Ray* R )
{
    const CollisionResult Res = this->Trace( R );
    if ( !Res.Hit )
        return float3( );

	const Material Mat = Res.HitObject->Material;

	if ( Res.HitObject->IsLightSource( ) && R->Depth == 0 )
		return Mat.Radiance;

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
			R->Depth = 0;
			break;
	}

	return Mat.Radiance + Mat.Color * Radiance<depth + 1>( R ) * CalculateBDRF( Mat.Type, Res, R );
}

template<>
__device__ float3 Raytracer::Radiance<5>( Ray* R )
{
	return float3( );
}