#pragma once

#include <array>
#include <cstdint>

#include "vk_mem_alloc.h"
#include "vulkan/vulkan.hpp"

#include <rhi/Barrier.h>
#include <rhi/BindGroup.h>
#include <rhi/BufferDesc.h>
#include <rhi/Diagnostics.h>
#include <rhi/Pipeline.h>
#include <rhi/Rendering.h>
#include <rhi/RhiTypes.h>
#include <rhi/SamplerDesc.h>
#include <rhi/TextureDesc.h>
#include <rhi/TextureViewDesc.h>

/**
 * Every mapping between the RHI's neutral vocabulary and Vulkan lives here and
 * nowhere else (plan §3). A conversion written inline at a call site is the
 * first step back towards Vulkan types leaking across the boundary, and it puts
 * the mapping somewhere the tests cannot reach it.
 *
 * Two directions, deliberately not symmetric:
 *
 *   ToVk   exists for every neutral enum. Implemented as a switch with no
 *          `default:` label, so adding an enumerator without a mapping fails
 *          the build rather than silently falling through. The throw sits
 *          after the switch instead.
 *
 *   FromVk exists only where the mapping is one-to-one — where each neutral
 *          value has its own Vulkan value and no two share one. It is derived
 *          from ToVk by searching the enumerator list rather than being a
 *          second hand-written table, so the two cannot disagree; the failure
 *          mode of duplicated tables is a typo in the copy that only shows up
 *          on one code path.
 *
 * The types with no FromVk are PipelineStage, AccessFlags, TextureLayout,
 * BufferUsage, TextureUsage and TextureAspect. Each fails the one-to-one test
 * in one of two ways:
 *
 *   * A single neutral flag can expand to several Vulkan flags. PipelineStage::
 *     DepthStencil is both the early and the late fragment-test stage, so there
 *     is no single Vulkan value to map back from.
 *   * Two neutral values can share one Vulkan value. TextureLayout::Common and
 *     ::UnorderedAccess are both eGeneral, so a reverse mapping would have to
 *     pick one and would be wrong half the time.
 *
 * Nothing needs the reverse for these: they describe work being handed to the
 * backend, never results coming back from it.
 */
namespace Hikari::Rhi::Vulkan
{
/**
 * Neutral load and store ops to Vulkan's.
 *
 * Switches without a default so that a new enumerator fails the build here
 * rather than silently mapping to whatever came first (plan D11's ratchet,
 * applied to the same problem).
 */
/**
 * Neutral shader stages to Vulkan's. Throws on an empty set: a binding or a
 * push constant range visible to no stage is a caller mistake rather than a
 * degenerate case worth expressing.
 */
vk::ShaderStageFlags ToVk(ShaderStage stages);

vk::CullModeFlags ToVk(CullMode mode);

/**
 * Neutral texture usage to the format features a device must advertise for it.
 * Separate from the usage-to-VkImageUsageFlags mapping: what a format must
 * support and what an image is created with are different questions with
 * different answers.
 */
vk::FormatFeatureFlags ToVkFormatFeatures(TextureUsage usage);

vk::AttachmentLoadOp ToVkLoadOp(LoadOp op);
vk::AttachmentStoreOp ToVkStoreOp(StoreOp op);

/**
 * VMA splits "where the memory lives" across a usage enum and a set of
 * allocation flags, so a neutral MemoryAccess converts to a pair rather than to
 * a single value. D3D12's equivalent is one heap type.
 */
struct VmaMemoryParams
{
    VmaMemoryUsage Usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocationCreateFlags Flags = 0;

    bool operator==(const VmaMemoryParams&) const = default;
};

// --- One-to-one: ToVk and FromVk ---

vk::Format ToVk(Format format);
Format FromVk(vk::Format format);

vk::SampleCountFlagBits ToVk(SampleCount samples);
SampleCount FromVk(vk::SampleCountFlagBits samples);

vk::PresentModeKHR ToVk(PresentMode mode);
PresentMode FromVk(vk::PresentModeKHR mode);

vk::ImageType ToVk(TextureDimension dimension);
TextureDimension FromVk(vk::ImageType imageType);

vk::ImageViewType ToVk(TextureViewDimension dimension);
TextureViewDimension FromVk(vk::ImageViewType viewType);

vk::Filter ToVk(Filter filter);
Filter FromVk(vk::Filter filter);

vk::SamplerMipmapMode ToVk(MipmapMode mode);
MipmapMode FromVk(vk::SamplerMipmapMode mode);

vk::SamplerAddressMode ToVk(AddressMode mode);
AddressMode FromVk(vk::SamplerAddressMode mode);

vk::CompareOp ToVk(CompareOp op);
CompareOp FromVk(vk::CompareOp op);

vk::BorderColor ToVk(BorderColor color);
BorderColor FromVk(vk::BorderColor color);

VmaMemoryParams ToVk(MemoryAccess access);
MemoryAccess FromVk(const VmaMemoryParams& params);

/**
 * Diagnostic severity, both ways. This is the one pair where FromVk is
 * many-to-one and exists anyway: Vulkan's eVerbose and eInfo both collapse to
 * Info, because the neutral scale deliberately has no verbose tier and nothing
 * treats the two differently. That makes ToVk(FromVk(eVerbose)) == eInfo rather
 * than eVerbose, which is intended rather than a lossy accident — so the
 * round-trip test asserts it in the one direction that holds.
 *
 * ToVk is needed because the severity is also used as a *threshold*: the
 * specification orders the enumerator values verbose < info < warning < error,
 * so a `>=` against the converted minimum filters messages correctly.
 */
vk::DebugUtilsMessageSeverityFlagBitsEXT ToVk(DiagnosticSeverity severity);
DiagnosticSeverity FromVk(vk::DebugUtilsMessageSeverityFlagBitsEXT severity);

/**
 * Whether a queue family advertising `familyCapabilities` (from
 * VkQueueFamilyProperties::queueFlags) can serve `role`.
 *
 * A predicate rather than a "role -> required bit" mapping, because the test is
 * not a bit comparison a caller could be trusted to write:
 *
 *   * Copy is satisfied by eTransfer, but *also* by eGraphics or eCompute. The
 *     spec makes reporting eTransfer optional on a family that already
 *     advertises graphics or compute, since such a family can always perform
 *     transfers. Testing for eTransfer alone would reject a graphics family
 *     that copies perfectly well, on any driver taking the spec up on that
 *     option.
 *   * The test is therefore "any of these bits", not "all of them". Exposing
 *     the mask instead would invite `HasAll(flags, mask)`, which for Copy would
 *     demand a family be graphics *and* compute *and* transfer — rejecting the
 *     dedicated transfer family a device's DMA engine appears as.
 *
 * One family can satisfy several roles, and normally does: a universal family
 * satisfies all three, which is why there is no inverse asking "which role is
 * this family". Whether a family is *dedicated* to a role is the narrower,
 * separate test — supports the role but not Graphics — and belongs to
 * SelectQueueFamilies rather than here.
 */
bool FamilySupports(vk::QueueFlags familyCapabilities, QueueType role);

// --- One-way: ToVk only, see the comment above ---

vk::PipelineStageFlags2 ToVk(PipelineStage stages);
vk::AccessFlags2 ToVk(AccessFlags access);
vk::ImageLayout ToVk(TextureLayout layout);
vk::BufferUsageFlags ToVk(BufferUsage usage);
vk::ImageUsageFlags ToVk(TextureUsage usage);
vk::ImageAspectFlags ToVk(TextureAspect aspect);
} // namespace Hikari::Rhi::Vulkan
