#pragma once

#include <cstdint>
#include <span>
#include <string>

#include <rhi/Handles.h>
#include <rhi/RhiTypes.h>

namespace Hikari::Rhi
{
/**
 * What kind of resource a slot holds.
 *
 * Deliberately three values. There is no combined texture-and-sampler kind:
 * D3D12 keeps samplers in a heap of their own, capped separately, and cannot
 * express one descriptor holding both (plan D22). A backend would have to
 * decompose such a binding rather than translate it, which is the lowest common
 * denominator D17 refuses elsewhere.
 *
 * Growing this list is how the binding model grows, and it is meant to be hard:
 * each backend switches over it without a default, so a new kind fails the build
 * until every backend maps it, and the layout inventory test fails until someone
 * says why the model needed to change (plan D21).
 */
enum class BindingType : uint8_t
{
    UniformBuffer,

    /**
     * Read-only, and named the way the shaders are: HLSL spells this Texture2D
     * and its read-write counterpart RWTexture2D, so the plain name meaning the
     * read-only one is the convention anyone reading these shaders already has.
     * Vulkan's "sampled image" is the term this deliberately does not use (D13).
     */
    Texture,

    /**
     * Read-write, and named for D3D12's unordered access view -- which is also
     * how the barrier vocabulary already spells it (TextureLayout::UnorderedAccess).
     * HLSL writes RWTexture2D; Vulkan calls the descriptor a storage image, and
     * that is the term this deliberately does not use (D13).
     */
    UnorderedAccessTexture,

    Sampler,
};

/**
 * Which shader stages can see a binding.
 *
 * Pixel rather than Fragment, per D13. Compute is here because the depth bind
 * group is read by the cloud dispatch as well as by pixel shaders, so stage
 * visibility is not something a graphics-only assumption can cover.
 */
enum class ShaderStage : uint32_t
{
    None = 0,
    Vertex = 1 << 0,
    Pixel = 1 << 1,
    Compute = 1 << 2,
};
RHI_DEFINE_FLAG_OPERATORS(ShaderStage)

struct BindGroupLayoutBinding
{
    uint32_t Slot = 0u;
    BindingType Type = BindingType::UniformBuffer;
    ShaderStage Visibility = ShaderStage::None;

    /**
     * The slot may be left empty, and the shader must then not read it.
     *
     * What a material set needs: a mesh with no normal map leaves that slot
     * unfilled rather than binding a dummy texture, and the shader branches on a
     * push constant instead. Vulkan spells the permission
     * VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT and requires the feature bit to
     * match; D3D12 gets there differently, since a descriptor a shader does not
     * access need not be valid in the first place.
     *
     * Off by default, because leaving a slot empty that the shader *does* read
     * is undefined behaviour on both APIs rather than an error either reports.
     */
    bool bOptional = false;
};

struct BindGroupLayoutDesc
{
    std::span<const BindGroupLayoutBinding> Bindings{};

    std::string DebugName;
};

/**
 * One slot's contents. Exactly one handle is read, chosen by Type, and the
 * others are ignored -- a tagged union spelled as a struct, because the
 * alternative is three parallel spans the caller has to keep in step.
 */
struct BindGroupBinding
{
    uint32_t Slot = 0u;
    BindingType Type = BindingType::UniformBuffer;

    BufferHandle Buffer{};
    TextureViewHandle View{};
    SamplerHandle Sampler{};
};

struct BindGroupDesc
{
    BindGroupLayoutHandle Layout{};
    std::span<const BindGroupBinding> Bindings{};

    std::string DebugName;
};
} // namespace Hikari::Rhi
