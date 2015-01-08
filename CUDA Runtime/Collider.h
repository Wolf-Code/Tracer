class Collider
{
public:
    __device__ static CollisionResult& Collide( Ray&, Object* );
    __device__ static CollisionResult& SphereCollision( Ray&, Object* );
    __device__ static CollisionResult& PlaneCollision( Ray&, Object* );
};

__device__ CollisionResult& Collider::Collide( Ray& R, Object* Obj )
{
	CollisionResult Res;
	Res.Hit = false;

    switch ( Obj->Type )
    {
        case ObjectType::SphereType:
            Res = SphereCollision( R, Obj );
			break;

        case ObjectType::PlaneType:
            Res = PlaneCollision( R, Obj );
			break;
    }

	Res.Ray = &R;
	return Res;
}

__device__ CollisionResult& Collider::SphereCollision( Ray& R, Object* Obj )
{
    CollisionResult Res;
    Res.Hit = false;

    SphereObject* Sphere = &Obj->Sphere;

    float A = VectorMath::Dot( R.Direction, R.Direction );
    float B = 2 * VectorMath::Dot( R.Direction, R.Start - Sphere->Position );
    float C = VectorMath::Dot( R.Start - Sphere->Position, R.Start - Sphere->Position ) - ( Sphere->Radius * Sphere->Radius );

    float Discriminant = B * B - 4 * A * C;
    if ( Discriminant < 0 )
        return Res;

    float DiscriminantSqrt = sqrt( Discriminant );
    float Q;
    if ( B < 0 )
        Q = ( -B - DiscriminantSqrt ) / 2.0;
    else
        Q = ( -B + DiscriminantSqrt ) / 2.0;

    float T0 = Q / A;
    float T1 = C / Q;

    if ( T0 > T1 )
    {
        float TempT0 = T0;
        T0 = T1;
        T1 = TempT0;
    }

    // Sphere is behind the ray's start position.
    if ( T1 < 0 )
        return Res;

    Res.Distance = T0 < 0 ? T1 : T0;
    Res.Hit = true;
    Res.Position = R.Start + R.Direction * Res.Distance;
    Res.Normal = VectorMath::Normalized( Res.Position - Sphere->Position );
    Res.HitObject = Obj;

    return Res;
}

__device__ CollisionResult& Collider::PlaneCollision( Ray& R, Object* Obj )
{
    CollisionResult Res;
    Res.Hit = false;

    PlaneObject* Plane = &Obj->Plane;
    float Div = VectorMath::Dot( Plane->Normal, R.Direction );
    if ( Div == 0 )
        return Res;

    float Distance = -( VectorMath::Dot( Plane->Normal, R.Start ) + Plane->Offset ) / Div;
    if ( Distance < 0 )
        return Res;

    Res.Hit = true;
    Res.HitObject = Obj;
    Res.Distance = Distance;
    Res.Normal = Plane->Normal;
    Res.Position = R.Start + R.Direction * Distance;

    return Res;
}