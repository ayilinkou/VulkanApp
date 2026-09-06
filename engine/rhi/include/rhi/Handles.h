#pragma once

#include <core/Handle.h>

/**
 * The identities that cross the RHI boundary. A caller holds one of these
 * rather than a pointer to a backend object, so nothing outside the module
 * needs to know that a texture is a VkImage plus a VmaAllocation (plan D2).
 *
 * Each is a distinct type, so a buffer handle cannot be passed where a texture
 * handle is expected. The tags are declared inline and never defined — they
 * exist only to separate the types.
 */
namespace Hikari::Rhi
{
using BufferHandle = Core::Handle<struct BufferTag>;
using TextureHandle = Core::Handle<struct TextureTag>;
using TextureViewHandle = Core::Handle<struct TextureViewTag>;
using SamplerHandle = Core::Handle<struct SamplerTag>;

/**
 * Paired with a uint64_t value at every use site rather than being waited on
 * by itself: D3D12 has exactly one synchronization primitive, ID3D12Fence with
 * a monotonically increasing value, and Vulkan's timeline semaphore matches it.
 *
 * IDevice creates, waits on and destroys these, and a submission names one to
 * signal. The upload context still waits on a fence it owns privately.
 */
using FenceHandle = Core::Handle<struct FenceTag>;

/**
 * A GPU-to-GPU ordering point with no value attached, and the counterpart to
 * FenceHandle rather than a lesser version of it: presentation is the one place
 * both APIs still order work by a single-shot object the caller never resets.
 *
 * Only IPresentTarget produces one. The target owns the object, decides how many
 * there are and when they are recycled; a handle is how a caller names one for
 * long enough to wait on or signal it in its own submit. Nothing else in the RHI
 * hands out a semaphore, and nothing should — a caller that wants ordering
 * against RHI-owned work wants a fence and a value.
 */
using SemaphoreHandle = Core::Handle<struct SemaphoreTag>;

/**
 * The shape of a bind group: which slots it has, of what kind, visible to which
 * shader stages. Immutable once created, and shared by every group built to it.
 */
using BindGroupLayoutHandle = Core::Handle<struct BindGroupLayoutTag>;

/**
 * A set of resources bound together, built to a layout. Immutable (plan D20):
 * changing what it points at means creating another and destroying this one.
 */
using BindGroupHandle = Core::Handle<struct BindGroupTag>;

/** Bind group layouts plus push constant ranges: a VkPipelineLayout, a root signature. */
using PipelineLayoutHandle = Core::Handle<struct PipelineLayoutTag>;

/** Compiled shader bytes. The engine loads them; the device makes a module of them (D24). */
using ShaderModuleHandle = Core::Handle<struct ShaderModuleTag>;

using GraphicsPipelineHandle = Core::Handle<struct GraphicsPipelineTag>;
} // namespace Hikari::Rhi
