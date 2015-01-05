struct Object
{
	unsigned int ID;
    ObjectType Type;
    SphereObject Sphere;
    PlaneObject Plane;
    Material Material;
	__device__ bool IsLightSource( void );
};

__device__ bool Object::IsLightSource( void )
{
	return	this->Material.Radiance.x > 0 ||
			this->Material.Radiance.y > 0 ||
			this->Material.Radiance.z > 0;
}

__device__ bool operator==( Object Obj1, Object Obj2 )
{
	return Obj1.ID == Obj2.ID;
}