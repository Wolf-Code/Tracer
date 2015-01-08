#ifndef __VECTORMATH_H__
#define __VECTORMATH_H__
#include <math.h>

class VectorMath
{
public:
	__device__ static float Dot( const float3&, const float3& );
	__device__ static float Length( const float3& );
    __device__ static void Normalize( float3& );
	__device__ static float LargestComponent( const float3& );
	__device__ static float3 Normalized( const float3& );
	__device__ static float3 MakeVector( const float, const float, const float );
	__device__ static float3 MakeVector( const float );
	__device__ static float3 Reflect( const float3&, const float3& );
	__device__ static float3 RandomCosineDirectionInSameDirection( const float3&, curandState* );
	__device__ static float3 RandomDirectionInSameDirection( const float3&, curandState* );
	__device__ static float3 RandomDirection( curandState* );
};

__device__ float3 operator/( const float3&, const float );
__device__ float3 operator*( const float3&, const float );
__device__ float3 operator*( const float, const float3& );
__device__ float3 operator*( const float3&, const float3& );
__device__ float3 operator+( const float3&, const float3& );
__device__ void operator+=( float3&, const float3& );
__device__ void operator*=( float3&, const float3& );
__device__ float3 operator-( const float3&, const float3& );
__device__ bool operator==( const float3&, const float3& );

__device__ float VectorMath::Dot( const float3& Vector, const float3& Vector2 )
{
    return	Vector.x * Vector2.x + 
			Vector.y * Vector2.y + 
			Vector.z * Vector2.z;
}

__device__ float VectorMath::Length( const float3& Vector )
{
    return sqrt( VectorMath::Dot( Vector, Vector ) );
}

__device__ void VectorMath::Normalize( float3& Vector )
{
    float L = VectorMath::Length( Vector );
    Vector.x /= L;
    Vector.y /= L;
    Vector.z /= L;
}

__device__ float VectorMath::LargestComponent( const float3& Vector )
{
	return fmaxf( fmaxf( Vector.x, Vector.y ), Vector.z );
}

__device__ float3 VectorMath::Normalized( const float3& Vector )
{
    float L = VectorMath::Length( Vector );

    return Vector / L;
}

__device__ float3 VectorMath::MakeVector( const float X, const float Y, const float Z )
{
    float3 New;
    New.x = X;
    New.y = Y;
    New.z = Z;

    return New;
}

__device__ float3 VectorMath::MakeVector( const float Value )
{
	return VectorMath::MakeVector( Value, Value, Value );
}


__device__ float3 VectorMath::Reflect( const float3& Vector, const float3& Normal )
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

__device__ float3 VectorMath::RandomDirectionInSameDirection( const float3& Direction, curandState* RandState )
{
	float3 Rand = VectorMath::RandomDirection( RandState );

	if ( VectorMath::Dot( Direction, Rand ) < 0 )
		Rand = Rand * -1;

	return Rand;
}

__device__ float3 VectorMath::RandomCosineDirectionInSameDirection( const float3& Direction, curandState* RandState )
{
	float3 Rand = VectorMath::RandomDirectionInSameDirection( Direction, RandState );

	Rand = VectorMath::Normalized( Rand + Direction );

	return Rand;
}




__device__ float3 operator*( const float3& Vector, const float Multiplier )
{
    return VectorMath::MakeVector(
        Vector.x * Multiplier,
        Vector.y * Multiplier,
        Vector.z * Multiplier
        );
}

__device__ float3 operator*( const float Multiplier, const float3& Vector )
{
    return Vector * Multiplier;
}

__device__ float3 operator*( const float3& Vector, const float3& Vector2 )
{
    return VectorMath::MakeVector(
        Vector.x * Vector2.x,
        Vector.y * Vector2.y,
        Vector.z * Vector2.z
        );
}

__device__ float3 operator/( const float3& Vector, const float Divider )
{
    return VectorMath::MakeVector(
        Vector.x / Divider,
        Vector.y / Divider,
        Vector.z / Divider
        );
}

__device__ float3 operator+( const float3& Vector, const float3& Vector2 )
{
	return VectorMath::MakeVector(
		Vector.x + Vector2.x,
		Vector.y + Vector2.y,
		Vector.z + Vector2.z
		);
}

__device__ void operator+=( float3& Vector, const float3& Vector2 )
{
	Vector.x += Vector2.x;
	Vector.y += Vector2.y;
	Vector.z += Vector2.z;
}

__device__ void operator*=( float3& Vector, const float3& Vector2 )
{
	Vector.x *= Vector2.x;
	Vector.y *= Vector2.y;
	Vector.z *= Vector2.z;
}

__device__ float3 operator-( const float3& Vector, const float3& Vector2 )
{
	return VectorMath::MakeVector(
		Vector.x - Vector2.x,
		Vector.y - Vector2.y,
		Vector.z - Vector2.z
		);
}

__device__ bool operator==( const float3& Vector, const float3& Vector2 )
{
	return Vector.x == Vector2.x && Vector.y == Vector2.y && Vector.z == Vector2.z;
}

#endif