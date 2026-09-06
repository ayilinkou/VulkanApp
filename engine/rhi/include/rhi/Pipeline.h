#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include <rhi/BindGroup.h>
#include <rhi/Handles.h>
#include <rhi/RhiTypes.h>
#include <rhi/SamplerDesc.h>

namespace Hikari::Rhi
{
/**
 * Constants pushed straight into the command list rather than through a buffer.
 *
 * 1:1 on both APIs -- Vulkan's push constants and D3D12's root constants -- and
 * declared on the layout rather than at the call site, because that is where
 * both put them. The stage matters as much as the size: this engine has ranges
 * on the pixel stage and on compute, and the two are not interchangeable.
 */
struct PushConstantRange
{
    ShaderStage Stages = ShaderStage::None;
    uint32_t Offset = 0u;
    uint32_t Size = 0u;
};

/**
 * What a pipeline can be handed: the bind group layouts, in slot order, and the
 * push constant ranges.
 *
 * A first-class object rather than something derived from a pipeline
 * description, because both APIs have one and its identity is what decides
 * whether bound groups survive a pipeline change (plan D23). Deriving it would
 * mean hashing descriptions to deduplicate, which is a cache in the layer whose
 * job is to not be subtly wrong on one backend.
 */
struct PipelineLayoutDesc
{
    std::span<const BindGroupLayoutHandle> BindGroupLayouts{};
    std::span<const PushConstantRange> PushConstantRanges{};

    std::string DebugName;
};

struct ShaderModuleDesc
{
    /**
     * Compiled bytes -- SPIR-V or DXIL, whichever this backend eats. The caller
     * loads them, because resolving a name to a file is a content question and
     * the RHI has no business owning a filesystem (plan D24). DeviceCaps says
     * which format to load.
     */
    std::span<const std::byte> Bytes{};

    std::string DebugName;
};

/** Which shader in a module to use, since one module may hold several. */
struct ShaderStageDesc
{
    ShaderModuleHandle Module{};
    std::string EntryPoint;
};

enum class VertexInputRate : uint8_t
{
    Vertex,
    Instance,
};

/** One vertex buffer's stride and stepping. `Slot` is the buffer it binds to. */
struct VertexBufferLayout
{
    uint32_t Slot = 0u;
    uint32_t Stride = 0u;
    VertexInputRate Rate = VertexInputRate::Vertex;
};

struct VertexAttribute
{
    uint32_t Location = 0u;
    uint32_t Slot = 0u;
    Rhi::Format AttributeFormat = Format::Undefined;
    uint32_t Offset = 0u;
};

enum class CullMode : uint8_t
{
    None,
    Front,
    Back,
};

/**
 * Curated to what this renderer uses, like Rhi::Format and BindingType. Adding
 * one means mapping it in every backend, which is the point.
 *
 * There is no CompareOp here: SamplerDesc already declares one covering all
 * eight comparisons, and depth testing wants the same vocabulary a sampler's
 * comparison does.
 */
enum class BlendFactor : uint8_t
{
    Zero,
    One,
    OneMinusSrcColor,
};

enum class BlendOp : uint8_t
{
    Add,
};

/** One render target's blending. Disabled blending ignores every other field. */
struct RenderTargetBlend
{
    bool bEnable = false;

    BlendFactor SrcColor = BlendFactor::One;
    BlendFactor DstColor = BlendFactor::Zero;
    BlendOp ColorOp = BlendOp::Add;

    BlendFactor SrcAlpha = BlendFactor::One;
    BlendFactor DstAlpha = BlendFactor::Zero;
    BlendOp AlphaOp = BlendOp::Add;
};

struct DepthState
{
    bool bTest = false;
    bool bWrite = false;
    CompareOp Compare = CompareOp::Less;
};

/**
 * A compute pipeline: a layout and one shader, and nothing else. There is no
 * fixed-function state to describe, which is why this is not a variation on the
 * graphics description.
 */
struct ComputePipelineDesc
{
    PipelineLayoutHandle Layout{};
    ShaderStageDesc Shader;

    std::string DebugName;
};

struct GraphicsPipelineDesc
{
    PipelineLayoutHandle Layout{};

    ShaderStageDesc VertexShader;
    ShaderStageDesc PixelShader;

    std::span<const VertexBufferLayout> VertexBuffers{};
    std::span<const VertexAttribute> VertexAttributes{};

    /**
     * The formats this pipeline renders into, which must match the attachments
     * a rendering scope is opened with. Vulkan checks this against
     * VkPipelineRenderingCreateInfo, D3D12 against the PSO's RTVFormats.
     */
    std::span<const Format> RenderTargetFormats{};
    std::span<const RenderTargetBlend> RenderTargetBlends{};

    /** Undefined when the pipeline has no depth attachment. */
    Format DepthFormat = Format::Undefined;
    DepthState Depth{};

    CullMode Cull = CullMode::None;

    /**
     * Cull mode is set per draw rather than baked in. Two-sided materials are a
     * per-batch property, and rebuilding a pipeline to flip one is not.
     */
    bool bDynamicCull = false;

    std::string DebugName;
};
} // namespace Hikari::Rhi
