#pragma once
#ifndef __CAMDATA_H__
#define __CAMDATA_H__

#include "CUDAIncluder.h"
#include "Ray.h"
#include "VectorMath.cuh"

struct CamData
{
    float3 Position;
    float3 Forward;
    float3 Right;
    float3 Up;
    float A;
    float Width;
    float Height;

	__device__ Ray GetRay( float X, float Y )
	{
		Ray R;
		R.Depth = 0;
		R.Start = this->Position;
		float3 Dir = this->Forward * this->A +
			this->Right * ( X / this->Width - 0.5f ) * ( this->Width / this->Height ) -
			this->Up * ( Y / this->Height - 0.5f );

		R.Direction = VectorMath::Normalized( Dir );
		return R;
	}
};

#endif