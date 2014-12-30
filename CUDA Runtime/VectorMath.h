#include <math.h>

class VectorMath
{
public:
    __device__ static float Dot( float3 Vector, float3 Vector2 );
    __device__ static float Length( float3 Vector );
    __device__ static void Normalize( float3* Vector );
    __device__ static float3 Normalized( float3 Vector );
};

__device__ float VectorMath::Dot( float3 Vector, float3 Vector2 )
{
    return Vector.x * Vector2.x + Vector.y * Vector2.y + Vector.z * Vector2.z;
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

__device__ float3 VectorMath::Normalized( float3 Vector )
{
    float L = VectorMath::Length( Vector );
    float3 New;
    New.x = Vector.x /= L;
    New.y = Vector.y /= L;
    New.z = Vector.z /= L;

    return New;
}

__device__ float3 operator*( float3 Vector, float Multiplier )
{
    float3 New;
    New.x = Vector.x * Multiplier;
    New.y = Vector.y * Multiplier;
    New.z = Vector.z * Multiplier;

    return New;
}

__device__ float3 operator*( float3 Vector, float3 Vector2 )
{
    float3 New;
    New.x = Vector.x * Vector2.x;
    New.y = Vector.y * Vector2.y;
    New.z = Vector.z * Vector2.z;

    return New;
}

__device__ float3 operator/( float3 Vector, float Divider )
{
    float3 New;
    New.x = Vector.x / Divider;
    New.y = Vector.y / Divider;
    New.z = Vector.z / Divider;

    return New;
}

__device__ float3 operator+( float3 Vector, float3 Vector2 )
{
    float3 New;
    New.x = Vector.x + Vector2.x;
    New.y = Vector.y + Vector2.y;
    New.z = Vector.z + Vector2.z;

    return New;
}

__device__ float3 operator+=( float3 Vector, float3 Vector2 )
{
    float3 New;
    New.x = Vector.x + Vector2.x;
    New.y = Vector.y + Vector2.y;
    New.z = Vector.z + Vector2.z;

    return New;
}

__device__ float3 operator-( float3 Vector, float3 Vector2 )
{
    float3 New;
    New.x = Vector.x - Vector2.x;
    New.y = Vector.y - Vector2.y;
    New.z = Vector.z - Vector2.z;

    return New;
}