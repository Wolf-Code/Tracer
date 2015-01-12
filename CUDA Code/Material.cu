#include "CUDAIncluder.h"
#include "Material.cuh"
#include "MaterialType.h"
#include "VectorMath.cuh"
#include "Defines.h"

__device__ float Material::BRDF( const float3& In, const float3& Out, const float3& Normal )
{
	if ( this->Type == Diffuse )
		return OneOverPI;

	if ( this->Type == Reflective )
	{
		if ( VectorMath::Reflect( In, Normal ) == Out )
			return 1.0f;
	}

	return 0.0f;
}

__device__ float Material::CosTheta( const float3& OutGoing, const float3& Normal )
{
	if ( this->Type == Reflective )
		return 1.0f;

	return abs( VectorMath::Dot( OutGoing, Normal ) );
}

__device__ float Material::PDF( void )
{
	if ( this->Type == Diffuse )
		return OneOverTwoPI;

	if ( this->Type == Reflective )
		return 1.0f;

	return 1.0f;
}