#pragma once

#include <array>
#include <cstdint>
#include <span>

#include <core/Extent2D.h>
#include <rhi/Handles.h>

namespace Hikari::Rhi
{
/**
 * What happens to an attachment's existing contents when rendering begins.
 *
 * Named for D3D12's beginning-access types rather than Vulkan's load ops (plan
 * D13), which is also the clearer pair of words: "preserve" and "discard" say
 * what happens to the contents, where "load" and "don't care" describe what the
 * implementation does about them.
 */
enum class LoadOp : uint8_t
{
    Preserve,
    Clear,
    Discard,
};

/**
 * What happens to an attachment's contents when rendering ends.
 *
 * Two values, because two is what both APIs agree on: Preserve is Vulkan's
 * STORE and D3D12's PRESERVE, Discard is DONT_CARE and DISCARD. There is
 * deliberately no third value for "this pass did not write it" -- that is what
 * DepthStencilTarget::bReadOnly says, and saying it twice would be two fields
 * that can disagree.
 */
enum class StoreOp : uint8_t
{
    Preserve,
    Discard,
};

struct Offset2D
{
    int32_t X = 0;
    int32_t Y = 0;
};

struct Rect2D
{
    Offset2D Offset{};
    Core::Extent2D Extent{};
};

/**
 * Depth range defaults to [0, 1] because that is what both APIs use and what
 * GLM is configured to produce (GLM_FORCE_DEPTH_ZERO_TO_ONE).
 *
 * Height is positive. The Y-flip Vulkan needs is applied once to the projection
 * matrix behind DeviceCaps::bFlipClipSpaceY (plan D10) rather than by the
 * negative-height viewport trick, which has no D3D12 equivalent.
 */
struct Viewport
{
    float X = 0.f;
    float Y = 0.f;
    float Width = 0.f;
    float Height = 0.f;
    float MinDepth = 0.f;
    float MaxDepth = 1.f;
};

/** One colour attachment. The layout it must be in is the backend's business. */
struct RenderTarget
{
    TextureViewHandle View;
    LoadOp Load = LoadOp::Preserve;
    StoreOp Store = StoreOp::Preserve;

    /** Read only when Load is Clear. RGBA, in that order, unpremultiplied. */
    std::array<float, 4> ClearColor{};
};

struct DepthStencilTarget
{
    TextureViewHandle View;
    LoadOp Load = LoadOp::Preserve;
    StoreOp Store = StoreOp::Preserve;

    /** Read only when Load is Clear. */
    float ClearDepth = 1.f;
    uint32_t ClearStencil = 0u;

    /**
     * The pass reads this depth buffer -- tests against it, samples it, or both
     * -- and never writes it. Both APIs need telling: Vulkan wants a read-only
     * layout and a store op of NONE, D3D12 wants D3D12_DSV_FLAG_READ_ONLY_DEPTH.
     * Getting it wrong is not a validation error on either; it is a hazard,
     * because a depth buffer bound writable and also sampled is a read-write
     * conflict on one resource.
     *
     * Store must stay Preserve when this is set. The contents do survive --
     * nothing wrote them -- so Preserve is the true answer, and Discard would be
     * claiming the pass threw away something it never touched.
     *
     * Note this is *not* D3D12's NO_ACCESS, which means the resource is neither
     * read nor written and must be paired with a NO_ACCESS beginning access. A
     * pass that reads depth is exactly the case NO_ACCESS excludes.
     */
    bool bReadOnly = false;
};

/**
 * One rendering scope: what is being drawn into, and over what area.
 *
 * Dynamic rendering only -- there are no render pass or subpass objects here,
 * now or later (plan D17). D3D12 has no equivalent of either, and inventing one
 * would mean building a lowest common denominator that neither API wants.
 */
struct RenderingDesc
{
    Rect2D RenderArea;

    std::span<const RenderTarget> RenderTargets{};

    /** Null when the pass has no depth or stencil attachment. */
    const DepthStencilTarget* pDepthStencil = nullptr;
};
} // namespace Hikari::Rhi
