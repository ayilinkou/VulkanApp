// Weighted Blended Order-Independent Transparency
// based on https://casual-effects.blogspot.com/2015/03/implemented-weighted-blended-order.html

#include "common.slangh"
#include "pbr.slangh"

struct MaterialPushConstant
{
	float4 Albedo;
	float Metallic;
	float Roughness;
	float AO;
	float Opacity;
	bool bHasAlbedoTex;
	bool bHasNormalTex;
	bool bHasMetallicRoughnessTex;
	bool bTwoSided;
};

struct VS_In
{
	// per vertex
	float3 Pos : POSITION0;
	float2 TexCoord : TEXCOORD0;
	float3 Normal : NORMAL0;
	float4 Tangent : TANGENT0;

	// per instance
	// transposed so that float4x4 constructor makes a row major matrix
	float4 TModelCol0 : POSITION1;
	float4 TModelCol1 : POSITION2;
	float4 TModelCol2 : POSITION3;
	float4 TModelCol3 : POSITION4;

	float4 TNormalCol0 : NORMAL1;
	float4 TNormalCol1 : NORMAL2;
	float4 TNormalCol2 : NORMAL3;

	float4x4 GetModelMatrix()
	{
		return float4x4(TModelCol0, TModelCol1, TModelCol2, TModelCol3);
	}

	float3x3 GetNormalMatrix()
	{
		return float3x3(TNormalCol0.xyz, TNormalCol1.xyz, TNormalCol2.xyz);
	}
};

struct VS_Out
{
	float4 Pos : SV_Position;
	float3 WorldPos : TEXCOORD0;
	float2 TexCoord : TEXCOORD1;
	float3 Normal : TEXCOORD2;
	float4 Tangent : TEXCOORD3;
};

[shader("vertex")]
VS_Out vertMain(VS_In v)
{
	VS_Out o;
	float4 worldPos = mul(float4(v.Pos, 1.f), v.GetModelMatrix());
	o.Pos = mul(mul(worldPos, globalBuffer.Camera.View), globalBuffer.Camera.Proj);
	o.WorldPos = worldPos.xyz;
	o.TexCoord = v.TexCoord;
	const float3x3 normalMatrix = v.GetNormalMatrix();
	o.Normal = mul(v.Normal, normalMatrix);
	o.Tangent = float4(mul(v.Tangent.xyz, normalMatrix), v.Tangent.w);
	return o;
}

// [[vk::binding(binding, set)]]
[[vk::binding(0, 1)]]
Texture2D<float4> albedoTex;
[[vk::binding(1, 1)]]
Texture2D<float4> normalTex;
[[vk::binding(2, 1)]]
Texture2D<float4> metallicRoughnessTex;
// One sampler for all three: they are read the same way, and a combined image
// sampler is not something the RHI's binding model has (D3D12 has no such
// descriptor).
[[vk::binding(3, 1)]]
SamplerState materialSampler;

[[vk::push_constant]]
MaterialPushConstant pcMatData;

struct FS_Out
{
	float4 Accum : SV_Target0;
	float Revealage : SV_Target1;
};

[shader("fragment")]
FS_Out fragMain(VS_Out f)
{
	FS_Out o;

	float3 albedo;
	if (pcMatData.bHasAlbedoTex)
		albedo = albedoTex.Sample(materialSampler, f.TexCoord).rgb;
	else
		albedo = pcMatData.Albedo.rgb;

	float metallic, roughness;
	if (pcMatData.bHasMetallicRoughnessTex)
	{
		metallic = metallicRoughnessTex.Sample(materialSampler, f.TexCoord).b;
		roughness = metallicRoughnessTex.Sample(materialSampler, f.TexCoord).g;
	}
	else
	{
		metallic = pcMatData.Metallic;
		roughness = pcMatData.Roughness;
	}

	float3 vertexNormal = normalize(f.Normal);

	float3 N;
	if (pcMatData.bHasNormalTex)
	{
		float3 tangentNormal = normalTex.Sample(materialSampler, f.TexCoord).xyz; // packed tangent space normal
		tangentNormal = tangentNormal * 2.f - float3(1.f); // unpacked tangent space

		float3 T = normalize(f.Tangent.xyz);
		T = normalize(T - vertexNormal * dot(T, vertexNormal)); // re-orthogonalise tangent
		float3 B = cross(vertexNormal, T) * f.Tangent.w; // w contains the handedness

		float3x3 TBN = float3x3(T, B, vertexNormal);
		N = normalize(mul(tangentNormal, TBN)); // world space normal
	}
	else
	{
		N = vertexNormal;
	}

	float3 camToFrag = normalize(f.WorldPos - globalBuffer.Camera.Pos);
	float viewNormalAlignment = dot(camToFrag, vertexNormal);
	bool bNormalNeedsFlip = pcMatData.bTwoSided && (viewNormalAlignment > 0.f);
	N = bNormalNeedsFlip ? -1.f * N : N;

	float3 V = -1.f * camToFrag; // view vector
	float3 Lo = float3(0.f, 0.f, 0.f);

	for (uint i = 0u; i < globalBuffer.Lights.PointLightCount; i++)
	{
		if (globalBuffer.Lights.PointLights[i].Intensity == 0.f)
			continue;

		float attenuation = CalcAttenuation(f.WorldPos, globalBuffer.Lights.PointLights[i].Pos);
		float3 Li = globalBuffer.Lights.PointLights[i].Color * globalBuffer.Lights.PointLights[i].Intensity * attenuation; // light radiance
		float3 L = normalize(globalBuffer.Lights.PointLights[i].Pos - f.WorldPos); // light vector

		Lo += CalcLight(N, V, L, Li, albedo.rgb, roughness, metallic);
	}

	for (uint i = 0u; i < globalBuffer.Lights.DirLightCount; i++)
	{
		if (globalBuffer.Lights.DirLights[i].Intensity == 0.f)
			continue;

		float3 Li = globalBuffer.Lights.DirLights[i].Color * globalBuffer.Lights.DirLights[i].Intensity; // light radiance
		float3 L = normalize(-globalBuffer.Lights.DirLights[i].Dir); // light vector

		Lo += CalcLight(N, V, L, Li, albedo.rgb, roughness, metallic);
	}

	// opacity should be influenced by fresnel
	float3 F0 = float3(0.04f);
	float NdotV = saturate(dot(N, V));
	float fresnel = SchlickFresnel(F0, NdotV).r;
	const float opacity = pcMatData.Opacity + (1.f - pcMatData.Opacity) * fresnel;

	// add reflections scaled by fresnel
	float3 skyColor = globalBuffer.SkyColor; // TODO: replace with environment map
	float3 ambientReflection = skyColor * fresnel;
	Lo += ambientReflection;

	float4 premultipliedReflect = float4(Lo * opacity, opacity);
	// TODO: add transmission
	const float3 transmit = float3(0.f, 0.f, 0.f);
	const float ndcZ = f.Pos.z;

    premultipliedReflect.a *= 1.f - clamp((transmit.r + transmit.g + transmit.b) * (1.f / 3.f), 0.f, 1.f);

    float a = min(1.f, premultipliedReflect.a) * 8.f + 0.01f;
	float b = -ndcZ * 0.95f + 1.f;

    // If your scene has a lot of content very close to the far plane,
    //   then include this line (one rsqrt instruction):
    //   b /= sqrt(1e4 * abs(csZ));
    float w    = clamp(a * a * a * 1e8f * b * b * b, 1e-2f, 3e2f);
    o.Accum     = premultipliedReflect * w;
    o.Revealage = premultipliedReflect.a;

	return o;
}
