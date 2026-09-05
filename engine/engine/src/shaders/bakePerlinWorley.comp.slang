[[vk::binding(0, 0)]] RWTexture3D<float> outputNoise;

struct BakeConstants
{
    uint resolution;
    uint worleyPointsPerCell; // typically 1, controls Worley cell density
};
[[vk::push_constant]] BakeConstants pc;

// ---- Tileable hash: wraps cell coordinates before hashing ----
float3 hash33(int3 p, uint res)
{
    // wrap to [0, res) so neighboring cells across the seam hash identically to a real wraparound
    int3 wrapped = ((p % int(res)) + int(res)) % int(res);
    float3 f = float3(wrapped) * 0.001f + float3(17.23f, 31.71f, 47.11f);
    f = frac(f * float3(443.897f, 441.423f, 437.195f));
    f += dot(f, f.yzx + 19.19f);
    return frac((f.xxy + f.yxx) * f.zyx);
}

// ---- Tileable gradient (Perlin) noise ----
float PerlinNoise3D(float3 p, uint res)
{
    int3 i0 = int3(floor(p));
    float3 f = frac(p);
    float3 u = f * f * (3.f - 2.f * f);

    float result = 0.f;
    float weightSum = 0.f;
    for (int z = 0; z <= 1; z++)
    for (int y = 0; y <= 1; y++)
    for (int x = 0; x <= 1; x++)
    {
        int3 corner = i0 + int3(x, y, z);
        float3 gradient = normalize(hash33(corner, res) * 2.f - 1.f);
        float3 offset = f - float3(x, y, z);
        float weight = (x == 0 ? (1 - u.x) : u.x) *
                       (y == 0 ? (1 - u.y) : u.y) *
                       (z == 0 ? (1 - u.z) : u.z);
        result += weight * dot(gradient, offset);
        weightSum += weight;
    }
    return result * 0.5f + 0.5f; // remap [-1,1] -> [0,1]
}

// ---- Tileable Worley (cellular) noise ----
float WorleyNoise3D(float3 p, uint res, uint pointsPerCell)
{
    int3 cell = int3(floor(p));
    float minDist = 1e9f;

    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++)
    {
        int3 neighborCell = cell + int3(x, y, z);
        for (uint k = 0; k < pointsPerCell; k++)
        {
            float3 randomOffset = hash33(neighborCell + int3(k * 97, k * 53, k * 71), res);
            float3 pointPos = float3(neighborCell) + randomOffset;
            float dist = length(pointPos - p);
            minDist = min(minDist, dist);
        }
    }
    return saturate(minDist); // cell size is 1 unit, so max meaningful distance is ~1
}

// ---- FBM wrappers ----
float PerlinFBM(float3 p, uint res, int octaves)
{
    float sum = 0.f, amplitude = 0.5f, freqMul = 1.f;
    for (int i = 0; i < octaves; i++)
    {
        sum += PerlinNoise3D(p * freqMul, res * uint(freqMul)) * amplitude;
        amplitude *= 0.5f;
        freqMul *= 2.f;
    }
    return sum;
}

float WorleyFBM(float3 p, uint res, uint pointsPerCell, int octaves)
{
    float sum = 0.f, amplitude = 0.5f, freqMul = 1.f;
    for (int i = 0; i < octaves; i++)
    {
        sum += (1.f - WorleyNoise3D(p * freqMul, res * uint(freqMul), pointsPerCell)) * amplitude;
        amplitude *= 0.5f;
        freqMul *= 2.f;
    }
    return sum;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId >= pc.resolution)) return;

    // frequency measured in "cells" across the texture — e.g. 4 means noise repeats every res/4 texels
    float3 p = float3(dispatchId);

    float perlin = PerlinFBM(p / 8.f, pc.resolution / 8u, 4);       // low frequency base shape
    float worley = WorleyFBM(p / 16.f, pc.resolution / 16u, pc.worleyPointsPerCell, 3); // erosion mask

    // classic Perlin-Worley remap (Schneider, "Real-Time Volumetric Cloudscapes"):
    // use inverted worley FBM as a floor, so perlin only shows through where worley carves cavities
    //float perlinWorley = saturate((perlin - (1.f - worley)) / max(worley, 0.0001f));
	float perlinWorley = saturate(perlin - (1.f - worley) * 0.5f); // erosion strength halved

    outputNoise[dispatchId] = perlinWorley;
}
