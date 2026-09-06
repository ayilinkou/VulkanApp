#include "common.slangh"
#include "pbr.slangh"

[[vk::binding(0, 1)]] Texture2D<float> depthTexture;
[[vk::binding(0, 2)]] RWTexture2D<float4> outputImage;
[[vk::binding(1, 2)]] Texture3D<float> perlinWorley;
// Separate from the texture: D3D12 keeps samplers in their own heap and has no
// combined descriptor, so the RHI's binding model has none either.
[[vk::binding(2, 2)]] SamplerState perlinWorleySampler;

struct PushConstants
{
	float3 windVelocity;
    float minHeight;
    float maxHeight;
    float coverage;
	float anisotropy;
	float boundaryDisplacement;
    uint viewStepCount;
	uint sunStepCount;
};
[[vk::push_constant]] PushConstants pc;

static const float kSlabEpsilon = 1e-5f;

void IntersectHeightSlab(float3 origin, float3 dir, float minH, float maxH, float margin,
                          out float tNear, out float tFar)
{
    // Near-horizontal dir never meaningfully crosses the slab planes; bail out
    // so t doesn't go to infinity. Callers treat tNear > tFar as "no hit".
    if (abs(dir.y) < kSlabEpsilon) { tNear = 1.f; tFar = 0.f; return; }
    float tMinPlane = ((minH - margin) - origin.y) / dir.y;
    float tMaxPlane = ((maxH + margin) - origin.y) / dir.y;
    tNear = min(tMinPlane, tMaxPlane);
    tFar  = max(tMinPlane, tMaxPlane);
}

float SampleDensity(float3 pos)
{
	const float windSpeed = 0.1f;

	float3 samplePos = pos * 0.00001f + pc.windVelocity * windSpeed * globalBuffer.Time;
    float n = perlinWorley.SampleLevel(perlinWorleySampler, samplePos, 0).r;

    // low-frequency 2D perturbation of the layer bounds, so the boundary isn't a flat plane
    float2 boundaryNoiseCoord = pos.xz * 0.0003f + pc.windVelocity.xz * globalBuffer.Time * 0.5f;
    float boundaryNoise = perlinWorley.SampleLevel(perlinWorleySampler, float3(boundaryNoiseCoord, 0.f), 0).r;
    float heightOffset = (boundaryNoise - 0.5f) * 2.f * pc.boundaryDisplacement;

    float localMin = pc.minHeight + heightOffset;
    float localMax = pc.maxHeight + heightOffset;

    float h = saturate((pos.y - localMin) / (localMax - localMin));
    float heightGradient = smoothstep(0.f, 0.3f, h) * smoothstep(1.f, 0.6f, h);

    return saturate(n - pc.coverage) * heightGradient;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchId.xy;
    uint2 outSize;
    outputImage.GetDimensions(outSize.x, outSize.y);
    if (pixel.x >= outSize.x || pixel.y >= outSize.y) return;

    float2 uv = (float2(pixel) + 0.5f) / float2(outSize);
    float2 ndc = uv * 2.f - 1.f;

    float4 nearPoint = mul(globalBuffer.Camera.InvViewProj, float4(ndc, 0.f, 1.f));
    float4 farPoint  = mul(globalBuffer.Camera.InvViewProj, float4(ndc, 1.f, 1.f));
    nearPoint /= nearPoint.w;
    farPoint  /= farPoint.w;

    float3 rayOrigin = globalBuffer.Camera.Pos;
    float3 rayDir = normalize(farPoint.xyz - nearPoint.xyz);

	uint2 depthSize;
	depthTexture.GetDimensions(depthSize.x, depthSize.y);
	uint2 depthCoord = uint2(uv * float2(depthSize));
	depthCoord = min(depthCoord, depthSize - 1u);

    float rawDepth = depthTexture.Load(int3(depthCoord, 0));
    float sceneDist;
    if (rawDepth >= 1.f)
	{
        sceneDist = 1e9f; // sky, no occlusion
    }
	else
	{
        float4 worldPos = mul(globalBuffer.Camera.InvViewProj, float4(ndc, rawDepth, 1.f));
        worldPos /= worldPos.w;
        sceneDist = length(worldPos.xyz - rayOrigin);
    }

    // Ray-slab intersection
	float tNear, tFar;
	IntersectHeightSlab(rayOrigin, rayDir, pc.minHeight, pc.maxHeight, pc.boundaryDisplacement, tNear, tFar);

    tNear = max(tNear, 0.f);
    tFar = min(tFar, sceneDist);

    if (tFar <= tNear || abs(rayDir.y) < kSlabEpsilon)
	{
        outputImage[pixel] = float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    float stepSize = (tFar - tNear) / float(pc.viewStepCount);
    float viewTransmittance = 1.f;
    float t = tNear;
	float3 scatteredLight = float3(0.f, 0.f, 0.f);
	const float3 sunDir = normalize(-globalBuffer.Lights.DirLights[0].Dir);
	const float3 sunColor = globalBuffer.Lights.DirLights[0].Color;
	const float sunIntensity = globalBuffer.Lights.DirLights[0].Intensity;
	const float extinction = 0.01f;
	const float cosTheta = dot(rayDir, sunDir);
	const float phase = HenyeyGreenstein(cosTheta, pc.anisotropy);
	const float3 ambientColor = float3(0.5f, 0.55f, 0.65f) * 0.3f;

	for (uint i = 0u; i < pc.viewStepCount; i++)
	{
        float3 samplePos = rayOrigin + rayDir * t;
        float density = SampleDensity(samplePos);

		float sunTNear, sunTFar;
        if (density > 0.f)
		{
            IntersectHeightSlab(samplePos, sunDir, pc.minHeight, pc.maxHeight, pc.boundaryDisplacement, sunTNear, sunTFar);
            sunTNear = max(sunTNear, 0.f);

            float sunStepSize = (sunTFar - sunTNear) / float(pc.sunStepCount);
            float sunT = sunTNear;

			float sigmaT = density * extinction;
			float sunTransmittance = 1.f;

			for (uint j = 0u; j < pc.sunStepCount; j++)
			{
				float3 sunSamplePos = samplePos + sunDir * sunT;
				float sunDensity = SampleDensity(sunSamplePos);

				if (sunDensity > 0.f)
				{
					sunTransmittance *= exp(-sunDensity * sunStepSize * extinction);
				}
				sunT += sunStepSize;
			}

            if (globalBuffer.Lights.DirLightCount > 0)
            {
		    	float3 sampleLight = (sunColor * sunIntensity * sunTransmittance * sigmaT * phase) + (ambientColor * sigmaT);
		    	scatteredLight += sampleLight * viewTransmittance * stepSize;
            }
            viewTransmittance *= exp(-sigmaT * stepSize);
        }

        if (viewTransmittance < 0.01f) break;
        t += stepSize;
    }

    outputImage[pixel] = float4(scatteredLight, 1.f - viewTransmittance);
}
