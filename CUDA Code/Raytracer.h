#pragma once
#ifndef __RAYTRACER_H__
#define __RAYTRACER_H__

#include "CUDAIncluder.h"
#include "VectorMath.cuh"
#include "MaterialType.h"
#include "Material.cuh"
#include "ObjectType.h"
#include "Ray.h"

#include "SphereObject.h"
#include "PlaneObject.h"
#include "MeshObject.h"
#include "Object.h"
#include "CollisionResult.h"
#include "Collider.h"
#include "CamData.cuh"
#include "Defines.h"

class Raytracer
{
public:
	__device__ Raytracer( Object*, unsigned int, unsigned int*, unsigned int, curandState* );
	__device__ CollisionResult Trace( Ray& );
	__device__ Object* GetRandomLight( );
	__device__ float3 ShadowRay( const CollisionResult& );
	__device__ float3 RadianceIterative( unsigned int, Ray& );
	__device__ float3 LEnvironment( Ray& );

private:
	curandState* RandState;
	Object* Objects;
	unsigned int ObjectCount;
	unsigned int* Lights;
	unsigned int LightCount;
};

__device__ Raytracer::Raytracer( Object* Objects, unsigned int ObjectCount, unsigned int* Lights, unsigned int LightCount, curandState* RandState )
	:Objects( Objects ), ObjectCount( ObjectCount ), Lights( Lights ), LightCount( LightCount ), RandState( RandState ){ }

__device__ CollisionResult Raytracer::Trace( Ray& R )
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

__device__ float3 Raytracer::LEnvironment( Ray& Ray )
{
	return float3( );
}

__device__ Object* Raytracer::GetRandomLight( )
{
	unsigned int ID = ( unsigned int )roundf( curand_uniform( RandState ) * ( LightCount - 1 ) );
	return &Objects[ Lights[ ID ] ];
}

__device__ float3 Raytracer::ShadowRay( const CollisionResult& Result )
{
	Object* L = Raytracer::GetRandomLight( );

	const float3 Start = Result.Position + Result.Normal * Bias;

	const float3 LightSamplePos = L->Sphere.RandomPositionOnSphere( RandState );
	const float3 DirToSample = VectorMath::Normalized( LightSamplePos - Start );

	Ray R;
	R.Start = Start;
	R.Direction = DirToSample;

	const CollisionResult Res = this->Trace( R );

	if ( Res.HitObject->ID != L->ID )
		return this->LEnvironment( R );

	const float Dist = VectorMath::Length( Start - LightSamplePos );
	const float DistSquared = Dist * Dist;

	return ( L->Material.Radiance ) / DistSquared;
}

__device__ float3 Raytracer::RadianceIterative( unsigned int MaxDepth, Ray& R )
{
	float3 Val = VectorMath::MakeVector( 0.0f, 0.0f, 0.0f );
	float3 ThroughPut = VectorMath::MakeVector( 1, 1, 1 );
	bool PrimaryRay = true;

	while ( R.Depth < MaxDepth )
	{
		const CollisionResult Res = this->Trace( R );

		if ( !Res.Hit )
			return Val + this->LEnvironment( R ) * ThroughPut;

		Material& Mat = Res.HitObject->Material;
		
		if ( Res.HitObject->IsLightSource( ) )
		{
			if ( PrimaryRay )
			{
				float3 Norm = VectorMath::Normalized( Mat.Radiance );
				return Norm / VectorMath::LargestComponent( Norm ) * ThroughPut;
			}

			return Val;
		}

		float3 Shadow = this->ShadowRay( Res );

		float3 NextDirection;
		if ( Mat.Type == Reflective )
		{
			NextDirection = VectorMath::Reflect( R.Direction, Res.Normal );
			Shadow = float3( );
		}
		else
		{
			NextDirection = VectorMath::RandomDirectionInSameDirection( Res.Normal, RandState );
			PrimaryRay = false;
		}

		const float BDRF = Mat.BRDF( R.Direction, NextDirection, Res.Normal );
		const float cos_theta = Mat.CosTheta( NextDirection, Res.Normal );
		const float PDF = Mat.PDF( );
		const float3 Mul = Mat.Color * ( BDRF * cos_theta ) / PDF;

		Val += Shadow * Mul * ThroughPut;
		ThroughPut *= Mul;

		if ( VectorMath::LargestComponent( ThroughPut ) < Bias || curand_uniform( this->RandState ) > BDRF )
			break;
		else
			ThroughPut = ThroughPut / BDRF;

		R.Depth++;
		R.Start = Res.Position + Res.Normal * Bias;
		R.Direction = NextDirection;
	}

	return Val;
}

#endif