#ifndef __VECTORMATH_H__
#define __VECTORMATH_H__
#include <math.h>

class VectorMath
{
public:
    __device__ static float Dot( float3, float3 );
    __device__ static float Length( float3 );
    __device__ static void Normalize( float3* );
	__device__ static float LargestComponent( float3* );
    __device__ static float3 Normalized( float3 );
    __device__ static float3 MakeVector( float, float, float );
    __device__ static float3 Reflect( float3, float3 );
	__device__ static float3 RandomCosineDirectionInSameDirection( float3, curandState* );
	__device__ static float3 RandomDirectionInSameDirection( float3, curandState* );
	__device__ static float3 RandomDirection( curandState* );
};

__device__ float3 operator/( float3, float );
__device__ float3 operator*( float3, float );
__device__ float3 operator*( float, float3 );
__device__ float3 operator*( float3, float3 );
__device__ float3 operator+( float3, float3 );
__device__ void operator+=( float3&, float3 );
__device__ float3 operator-( float3, float3 );
__device__ bool operator==( float3, float3 );

__device__ float VectorMath::Dot( float3 Vector, float3 Vector2 )
{
    return	Vector.x * Vector2.x + 
			Vector.y * Vector2.y + 
			Vector.z * Vector2.z;
}

__device__ float VectorMath::Length( float3 Vector )
{
    return sqrt( VectorMath::Dot( Vector, Vector ) );
}

__device__ void VectorMath::Normalize( float3* Vector )
{
    float L = VectorMath::Length( *Vector );
    Vector->x /= L;
    Vector->y /= L;
    Vector->z /= L;
}

__device__ float VectorMath::LargestComponent( float3* Vector )
{
	return fmaxf( fmaxf( Vector->x, Vector->y ), Vector->z );
}

__device__ float3 VectorMath::Normalized( float3 Vector )
{
    float L = VectorMath::Length( Vector );

    return Vector / L;
}

__device__ float3 VectorMath::MakeVector( float X, float Y, float Z )
{
    float3 New;
    New.x = X;
    New.y = Y;
    New.z = Z;

    return New;
}

__device__ float3 VectorMath::Reflect( float3 Vector, float3 Normal )
{
	return Vector - 2.0f * VectorMath::Dot( Vector, Normal ) * Normal;
}

__device__ float3 VectorMath::RandomDirection( curandState* RandState )
{
	float3 Rand = VectorMath::MakeVector( curand_uniform( RandState ) * 2.0f - 1.0f,
										  curand_uniform( RandState ) * 2.0f - 1.0f,
										  curand_uniform( RandState ) * 2.0f - 1.0f );

	Rand = VectorMath::Normalized( Rand );

	return Rand;
}

__device__ float3 VectorMath::RandomDirectionInSameDirection( float3 Direction, curandState* RandState )
{
	float3 Rand = VectorMath::RandomDirection( RandState );

	if ( VectorMath::Dot( Direction, Rand ) < 0 )
		Rand = Rand * -1;

	return Rand;
}

__device__ float3 VectorMath::RandomCosineDirectionInSameDirection( float3 Direction, curandState* RandState )
{
	float3 Rand = VectorMath::RandomDirectionInSameDirection( Direction, RandState );

	Rand = VectorMath::Normalized( Rand + Direction );

	return Rand;
}




__device__ float3 operator*( float3 Vector, float Multiplier )
{
    return VectorMath::MakeVector(
        Vector.x * Multiplier,
        Vector.y * Multiplier,
        Vector.z * Multiplier
        );
}

__device__ float3 operator*( float Multiplier, float3 Vector )
{
    return Vector * Multiplier;
}

__device__ float3 operator*( float3 Vector, float3 Vector2 )
{
    return VectorMath::MakeVector(
        Vector.x * Vector2.x,
        Vector.y * Vector2.y,
        Vector.z * Vector2.z
        );
}

__device__ float3 operator/( float3 Vector, float Divider )
{
    return VectorMath::MakeVector(
        Vector.x / Divider,
        Vector.y / Divider,
        Vector.z / Divider
        );
}

__device__ float3 operator+( float3 Vector, float3 Vector2 )
{
	return VectorMath::MakeVector(
		Vector.x + Vector2.x,
		Vector.y + Vector2.y,
		Vector.z + Vector2.z
		);
}

__device__ void operator+=( float3& Vector, float3 Vector2 )
{
	Vector.x += Vector2.x;
	Vector.y += Vector2.y;
	Vector.z += Vector2.z;
}

__device__ float3 operator-( float3 Vector, float3 Vector2 )
{
	return VectorMath::MakeVector(
		Vector.x - Vector2.x,
		Vector.y - Vector2.y,
		Vector.z - Vector2.z
		);
}

__device__ bool operator==( float3 Vector, float3 Vector2 )
{
	return Vector.x == Vector2.x && Vector.y == Vector2.y && Vector.z == Vector2.z;
}

#endif