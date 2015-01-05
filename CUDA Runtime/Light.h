#ifndef __LIGHT_H__
#define __LIGHT_H__

class Light
{
public:
	__device__ static float3 Radiance( float3, float3, float3 );
};

__device__ float3 Light::Radiance( float3 LightPosition, float3 SurfacePosition, float3 LightRadiance )
{
	float Dist = VectorMath::Length( LightPosition - SurfacePosition );
	return LightRadiance / ( Dist * Dist );
}

#endif

