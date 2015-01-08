struct Material
{
    float3 Color;
    float3 Radiance;
    MaterialType Type;
    float Glossyness;
	__device__ float3 BRDF( float3, float3, float3 );
	__device__ float CosTheta( float3, float3 );
	__device__ float PDF( void );
};

__device__ float3 Material::BRDF( float3 In, float3 Out, float3 Normal )
{
	if ( this->Type == Diffuse )
		return this->Color * OneOverPI;

	if ( this->Type == Reflective )
	{
		if ( VectorMath::Reflect( In, Normal ) == Out )
			return this->Color;
		else
			return float3( );
	}
}

__device__ float Material::CosTheta( float3 OutGoing, float3 Normal )
{
	return abs( VectorMath::Dot( OutGoing, Normal ) );
}

__device__ float Material::PDF( void )
{
	if ( this->Type == Diffuse )
		return OneOverTwoPI;

	if ( this->Type == Reflective )
		return 1.0f;
}