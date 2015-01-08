#define PI 3.1415926535
#define TwoPI PI * 2
#define OneOverPI 0.31830988618
#define OneOverTwoPI 0.1591549430
#define Bias 0.01


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
	__device__ Raytracer( Object*, unsigned int, Object*, unsigned int, curandState* );
	__device__ CollisionResult Trace( Ray* );
	__device__ float CalculateBDRF( MaterialType, CollisionResult, Ray* );
	__device__ Object* GetRandomLight( );
	__device__ float3 ShadowRay( CollisionResult* );
	__device__ float3 RadianceIterative( unsigned int, Ray* );
	__device__ float3 LEnvironment( Ray* );

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
	CollisionResult Res;
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

	float3 LightSamplePos = L->Sphere.RandomPositionOnSphere( RandState );
	float3 DirToSample = VectorMath::Normalized( LightSamplePos - Start );

	Ray R;
	R.Start = Start;
	R.Direction = DirToSample;

	const CollisionResult Res = this->Trace( &R );

	if ( Res.HitObject->ID != L->ID )
		return this->LEnvironment( &R );

	float Dist = VectorMath::Length( Start - LightSamplePos );
	float DistSquared = Dist * Dist;

	return L->Material.Radiance / DistSquared;
}

__device__ float3 Raytracer::RadianceIterative( unsigned int MaxDepth, Ray* R )
{
	float3 Val = VectorMath::MakeVector( 0.0f, 0.0f, 0.0f );
	float3 ThroughPut = VectorMath::MakeVector( 1, 1, 1 );
	int DepthSinceNonDiffuse = 0;

	while ( R->Depth < MaxDepth )
	{
		CollisionResult Res = this->Trace( R );

		if ( !Res.Hit )
			return Val + this->LEnvironment( R ) * ThroughPut;

		Material* Mat = &Res.HitObject->Material;
		
		if ( Res.HitObject->IsLightSource( ) )
		{
			if ( DepthSinceNonDiffuse == 0 )
				return Mat->Radiance;
			else
				return Val;
		}


		float3 NextDirection;
		if ( Mat->Type == Reflective )
		{
			NextDirection = VectorMath::Reflect( R->Direction, Res.Normal );
			DepthSinceNonDiffuse = 0;
		}
		else
			NextDirection = VectorMath::RandomDirectionInSameDirection( Res.Normal, RandState );

		float3 BDRF = Mat->BRDF( R->Direction, NextDirection, Res.Normal );
		float cos_theta = Mat->CosTheta( NextDirection, Res.Normal );
		float PDF = Mat->PDF( );
		float3 Mul = ( BDRF * cos_theta ) / PDF;
		float3 Shadow = this->ShadowRay( &Res );

		Val += Shadow * Mul * ThroughPut;

		ThroughPut *= Mul;
		if ( VectorMath::LargestComponent( ThroughPut ) < Bias )
			break;

		R->Depth++;
		R->Start = Res.Position + Res.Normal * Bias;
		R->Direction = NextDirection;
	}

	return Val;
}