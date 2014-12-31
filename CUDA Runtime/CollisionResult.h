struct CollisionResult
{
    bool Hit;
    float3 Position;
    float3 Normal;
    float Distance;
    Object* HitObject;
};