struct UniformBuffer
{
	float4x4 Model;
	float4x4 View;
	float4x4 Proj;
};

ConstantBuffer<UniformBuffer> ubo;

struct VS_In
{
	float2 Pos;
	float3 Color;
};

struct VS_Out
{
	float4 Pos : SV_Position;
	float3 Color;
};

[shader("vertex")]
VS_Out vertMain(VS_In v)
{
	VS_Out o;
	o.Pos = mul(ubo.Proj, mul(ubo.View, mul(ubo.Model, float4(v.Pos, 0.f, 1.f))));
	o.Color = v.Color;
	return o;
}

[shader("fragment")]
float4 fragMain(VS_Out f) : SV_Target
{
	return float4(f.Color, 1.f);
}
