#pragma once
#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "CUDAIncluder.h"
#include "MaterialType.h"

struct Material
{
    float3 Color;
    float3 Radiance;
    MaterialType Type;
    float Glossyness;
	__device__ float BRDF( const float3& In, const float3& Out, const float3& Normal );
	__device__ float CosTheta( const float3&, const float3& );
	__device__ float PDF( void );
};

#endif