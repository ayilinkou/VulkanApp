static float2 positions[3] = float2[](
	float2(0.f, -0.5f),
	float2(0.5f, 0.5f),
	float2(-0.5f, 0.5f)
);

static float3 colors[3] = float3[](
	float3(1.f, 0.f, 0.f),
	float3(0.f, 1.f, 0.f),
	float3(0.f, 0.f, 1.f)
);

struct VS_Out
{
	float4 Pos : SV_Position;
	float3 Color;
};

[shader("vertex")]
VS_Out vertMain(uint vid : SV_VertexID)
{
	VS_Out o;
	o.Pos = float4(positions[vid], 0.f, 1.f);
	o.Color = colors[vid];
	return o;
}

[shader("fragment")]
float4 fragMain(VS_Out f) : SV_Target
{
	return float4(f.Color, 1.f);
}
