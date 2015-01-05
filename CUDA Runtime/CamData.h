#ifndef __CAMDATA_H__
#define __CAMDATA_H__
#include "CUDAIncluder.h"
#include "Ray.h"

struct CamData
{
    float3 Position;
    float3 Forward;
    float3 Right;
    float3 Up;
    float A;
    float Width;
    float Height;

public:
    __device__ Ray GetRay( float, float );
};

__device__ Ray CamData::GetRay( float X, float Y )
{
	float WH = this->Width / this->Height;

	Ray R;
	R.Depth = 0;
	R.Start = this->Position;
	float3 Dir = this->Forward * this->A +
		this->Right * ( X / this->Width - 0.5f ) * WH -
		this->Up * ( Y / this->Height - 0.5f );

	R.Direction = VectorMath::Normalized( Dir );
	return R;
}

#endif