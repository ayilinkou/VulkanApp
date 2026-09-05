#include "common.slangh"
#include "pbr.slangh"

struct VS_In
{
	float2 Pos : POSITION0;
	float2 TexCoord : TEXCOORD0;
};

struct VS_Out
{
	float4 Pos : SV_Position;
	float2 TexCoord : TEXCOORD0;
};

[shader("vertex")]
VS_Out vertMain(VS_In v)
{
	VS_Out o;
	o.Pos = float4(v.Pos, 0.f, 1.f);
	o.TexCoord = v.TexCoord;
	return o;
}

[[vk::binding(0, 1)]] Texture2D<float4> opaqueTex;
[[vk::binding(1, 1)]] Texture2D<float4> accumTex;
[[vk::binding(2, 1)]] Texture2D<float> revealageTex;
[[vk::binding(3, 1)]] Sampler2D cloudsTex;

[shader("fragment")]
float4 fragMain(VS_Out f) : SV_Target
{
	int3 texel = int3((int2)f.Pos.xy, 0);
	float4 opaque = float4(opaqueTex.Load(texel).rgb, 1.f);
	float4 clouds = cloudsTex.Sample(f.TexCoord); // quarter res
	
	float3 composite;
	float3 srcColor = clouds.rgb * clouds.a + opaque.rgb * (1.f - clouds.a);
	float revealage = revealageTex.Load(texel);
	if (revealage == 1.f)
	{
		composite = srcColor;
	}
	else
	{
		float4 accum = accumTex.Load(texel);
		float4 accumAbs = abs(accum);
		float maxComponent = max(max(accumAbs.r, accumAbs.g), accumAbs.b);

		if (isinf(maxComponent))
			accum.rgb = float3(accum.a);

		float epsilon = 0.00001f;
		float3 averageColor = accum.rgb / max(accum.a, epsilon);

		composite = averageColor * (1.f - revealage) + srcColor * revealage;
	}

	composite = HillACES(composite); // tone mapping
	composite = pow(composite, float3(1.f / 2.2f)); // gamma correction

	return float4(composite, 1.f);
}
