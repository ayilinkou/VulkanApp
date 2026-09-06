#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "vulkan/vulkan_raii.hpp"

#include <rhi/BarrierPresets.h>
#include <rhi/ICommandList.h>
#include <rhi/IDevice.h>
#include <rhi/IPresentTarget.h>
#include <rhi/RhiTypes.h>

#include "vulkan/OffscreenTarget.h"

#include "GpuReadback.h"
#include "RhiTestFixture.h"
#include "ValidationGuard.h"

/**
 * The headless half of the presentation seam.
 *
 * A device created without presentation support has no surface, so
 * CreatePresentTarget hands back an OffscreenTarget instead of a swapchain.
 * That is the only way these cases reach it — the target's type is deliberately
 * not nameable from outside the module, which is also what makes these tests
 * worth having: they exercise the offscreen path through exactly the interface
 * the renderer uses, so anything they cover is covered for the renderer too.
 *
 * A swapchain cannot be tested here at all: it needs a surface, which needs a
 * window, which a test binary does not have. What that costs is covered in the
 * architecture plan's CI section, and is why the windowed path is still checked
 * by running the application.
 */
using namespace Hikari::Core;
using namespace Hikari::Rhi;

namespace
{
/**
 * The extent every case renders at. Non-square and not a power of two, so a
 * row-pitch or a width/height transposition shows up as garbage rather than as
 * a picture that happens to still be square.
 */
constexpr Extent2D kExtent{253u, 101u};

/**
 * Clear colours whose components are all exactly 0 or 1, so the readback can
 * assert exact bytes: any rounding an implementation applies converting a float
 * clear value to UNORM8 lands on 0 or 255 either way. Distinct per frame and
 * per channel, so both a stale frame and a swapped channel are visible.
 */
constexpr std::array<std::array<float, 4>, 3> kFrameColors{
    std::array<float, 4>{1.f, 0.f, 0.f, 1.f},
    std::array<float, 4>{0.f, 1.f, 0.f, 1.f},
    std::array<float, 4>{0.f, 0.f, 1.f, 1.f},
};

/**
 * One frame's command pool and buffer. A pool each rather than one reset
 * between frames, because the point of the loop below is to have every frame in
 * flight at once: resetting a pool whose buffer the GPU is still reading is
 * undefined behaviour, and it is exactly the overlap these cases exist to
 * exercise.
 */
/**
 * One frame's recording storage. An allocator each rather than one shared,
 * because these cases deliberately keep several frames in flight at once and an
 * allocator may not be reset while work recorded from it is still running.
 */
struct FrameCommands
{
    std::unique_ptr<ICommandAllocator> Allocator;
    ICommandList* List = nullptr;
};

FrameCommands MakeFrameCommands(IDevice& device)
{
    FrameCommands frame;
    frame.Allocator = device.CreateCommandAllocator(
        CommandAllocatorDesc{.Queue = QueueType::Graphics, .DebugName = "Present Test Frame"});
    frame.List = &frame.Allocator->Acquire();
    return frame;
}

/**
 * One clear: a colour and the rectangle of the image it covers. A render pass
 * clears its render area and nothing else, so a list of these paints a
 * deliberately non-uniform image without a shader or a vertex buffer — which is
 * what a stride check needs, since a solid colour looks the same however the
 * rows are laid out.
 */
struct ClearRect
{
    std::array<float, 4> Color{};
    Rect2D Area{};
};

Rect2D WholeImage(Extent2D extent)
{
    return Rect2D{.Extent = extent};
}

/**
 * Records `clears` in order into `acquired`, each through a dynamic-rendering
 * pass of its own — the same shape as the renderer's composite pass: acquire,
 * transition, render into the acquired view, transition to what comes next.
 *
 * It leaves the image in ShaderResource rather than Present. An offscreen image
 * is not presentable and never can be — VK_IMAGE_LAYOUT_PRESENT_SRC_KHR belongs
 * to VK_KHR_swapchain, which a device with no surface does not enable — so
 * ShaderResource is the finished state that matches the target's Sampled usage,
 * and the layout the readbacks below hand to ReadRenderedTexture.
 */
void RecordClears(ICommandList* list, const AcquiredImage& acquired,
                  std::span<const ClearRect> clears)
{
    list->Begin();
    list->Barrier(BarrierPresets::AcquiredImageToRenderTarget().On(acquired.Texture));

    for (size_t i = 0; i < clears.size(); i++)
    {
        if (i > 0u)
        {
            // Two passes writing the same attachment are not implicitly
            // ordered against each other. PreserveRenderTarget as written
            // describes a pass that *loads* what the last one wrote; here the
            // second pass clears instead, so the destination access is the
            // write rather than the read.
            TextureBarrier betweenPasses = BarrierPresets::PreserveRenderTarget();
            betweenPasses.DstAccess = AccessFlags::RenderTargetWrite;
            list->Barrier(betweenPasses.On(acquired.Texture));
        }

        // A rendering scope whose only content is the clear the load op performs.
        const std::array renderTargets{RenderTarget{
            .View = acquired.View, .Load = LoadOp::Clear, .ClearColor = clears[i].Color}};

        list->BeginRendering(
            RenderingDesc{.RenderArea = clears[i].Area, .RenderTargets = renderTargets});
        list->EndRendering();
    }

    list->Barrier(BarrierPresets::RenderTargetToShaderResource().On(acquired.Texture));
    list->End();
}

void RecordClearFrame(ICommandList* list, const AcquiredImage& acquired,
                      const std::array<float, 4>& color, Extent2D extent)
{
    const std::array clears{ClearRect{.Color = color, .Area = WholeImage(extent)}};
    RecordClears(list, acquired, clears);
}

/**
 * Submits `list` with the waits the acquire asked for and the signal the target
 * requires before Present will accept the image.
 */
void SubmitFrame(IDevice& device, ICommandList& list, std::span<const SemaphoreHandle> waitOn,
                 SemaphoreHandle signalOnComplete)
{
    ICommandList* pList = &list;
    device.Submit(SubmitDesc{.Queue = QueueType::Graphics,
                             .CommandLists = {&pList, 1u},
                             .WaitSemaphores = waitOn,
                             .SignalSemaphores = {&signalOnComplete, 1u}});
}

/**
 * The bytes a clear to `color` leaves in memory, in `format`'s channel order.
 * Written out rather than assumed, because getting it wrong is precisely the
 * mistake a readback is meant to catch — the renderer's screenshot writer has a
 * hardcoded BGRA swizzle for exactly this reason.
 * The target the device hands back, as the concrete type TakePendingSignal lives on.
 *
 * A downcast rather than a member on IPresentTarget: reading an image outside a
 * frame is a question only a target that owns its images can answer, so the
 * interface deliberately does not ask it (architecture plan §10.2). Doing it
 * through dynamic_cast rather than by constructing an OffscreenTarget directly
 * keeps the device's own choice under test — a device that started handing back
 * something else would fail here rather than silently testing a target the
 * renderer would never be given.
 */
Vulkan::OffscreenTarget& AsOffscreen(IPresentTarget& target)
{
    auto* pOffscreen = dynamic_cast<Vulkan::OffscreenTarget*>(&target);
    REQUIRE(pOffscreen != nullptr);
    return *pOffscreen;
}

std::array<std::byte, 4> ExpectedTexel(Format format, const std::array<float, 4>& color)
{
    const auto quantize = [](float value)
    { return static_cast<std::byte>(static_cast<unsigned char>(value * 255.f)); };

    if (format == Format::BGRA8Unorm)
        return {quantize(color[2]), quantize(color[1]), quantize(color[0]), quantize(color[3])};

    REQUIRE(format == Format::RGBA8Unorm);
    return {quantize(color[0]), quantize(color[1]), quantize(color[2]), quantize(color[3])};
}
} // namespace

TEST_CASE("A device with no surface hands back an offscreen present target", "[rhi][gpu][present]")
{
    IDevice& device = RhiTest::RequireDevice();
    const RhiTest::ValidationGuard guard(device);

    REQUIRE_FALSE(device.GetCaps().bPresentSupported);

    const std::unique_ptr<IPresentTarget> target =
        device.CreatePresentTarget(PresentTargetDesc{.Extent = kExtent, .FramesInFlight = 2u});
    REQUIRE(target != nullptr);

    // The extent is honoured exactly. A swapchain's is clamped to what the
    // surface allows; nothing clamps this one, which is what makes a headless
    // capture reproducible across machines.
    CHECK(target->GetExtent() == kExtent);
    CHECK(target->GetImageCount() == 2u);

    // Undefined would mean the target had no format it could name, which the
    // renderer would then hand to pipeline creation.
    CHECK(target->GetFormat() != Format::Undefined);
}

TEST_CASE("An offscreen acquire always succeeds and cycles its images", "[rhi][gpu][present]")
{
    IDevice& device = RhiTest::RequireDevice();
    const RhiTest::ValidationGuard guard(device);

    const std::unique_ptr<IPresentTarget> target =
        device.CreatePresentTarget(PresentTargetDesc{.Extent = kExtent, .FramesInFlight = 3u});

    REQUIRE(target->GetImageCount() == 3u);

    // Never asks to be recreated: there is no surface to go out of date. The
    // renderer's out-of-date branch is therefore dead in a headless run, rather
    // than something it has to be able to do without a window.
    for (uint32_t frame = 0u; frame < 7u; frame++)
    {
        const AcquiredImage acquired = target->Acquire();
        CHECK_FALSE(acquired.bNeedsRecreate);
        CHECK(acquired.Index == frame % 3u);
        CHECK(acquired.Texture.IsValid());
        CHECK(acquired.View.IsValid());

        // Nothing has been submitted, so no image has a render-complete signal
        // outstanding and there is nothing to wait on — including on the second
        // pass over the images.
        CHECK(acquired.WaitSemaphores.empty());
    }
}

/**
 * The step's headline check: three frames through the same acquire / render /
 * present sequence the windowed renderer runs, into a target with no window,
 * with the frames overlapping rather than being waited on one at a time.
 *
 * Two images and three frames is the smallest arrangement that reuses one, so
 * frame 2 has to wait on the render-complete semaphore frame 0 signalled. That
 * is both the real write-after-write dependency and the only thing that leaves
 * the semaphore unsignalled in time for frame 2 to signal it again — a target
 * that dropped it would fail here with a validation error rather than by
 * rendering something subtly wrong.
 */
TEST_CASE("Three overlapping frames render into an offscreen target", "[rhi][gpu][present]")
{
    IDevice& device = RhiTest::RequireDevice();
    const RhiTest::ValidationGuard guard(device);

    const std::unique_ptr<IPresentTarget> target =
        device.CreatePresentTarget(PresentTargetDesc{.Extent = kExtent, .FramesInFlight = 2u});
    REQUIRE(target->GetImageCount() == 2u);

    std::vector<FrameCommands> frames;
    frames.reserve(kFrameColors.size());
    for (size_t i = 0; i < kFrameColors.size(); i++)
        frames.push_back(MakeFrameCommands(device));

    std::array<TextureHandle, 2> imagesByIndex{};

    for (size_t frame = 0; frame < kFrameColors.size(); frame++)
    {
        const AcquiredImage acquired = target->Acquire();
        REQUIRE_FALSE(acquired.bNeedsRecreate);

        // Only the third frame reuses an image, and only it has a previous
        // write to wait for.
        CHECK(acquired.WaitSemaphores.size() == (frame < 2u ? 0u : 1u));

        imagesByIndex[acquired.Index] = acquired.Texture;

        RecordClearFrame(frames[frame].List, acquired, kFrameColors[frame], kExtent);
        SubmitFrame(device, *frames[frame].List, acquired.WaitSemaphores,
                    target->GetRenderCompleteSemaphore(acquired.Index));

        CHECK(target->Present(acquired.Index));
    }

    // No WaitIdle: each read waits on the render-complete semaphore its frame
    // signalled and fences its own copy, so the ordering it needs is ordering it
    // establishes. A stray WaitIdle here would hide a read that established
    // none — which is why the semaphore is passed in rather than looked up.

    // Image 0 was written by frames 0 and 2, image 1 by frame 1, so what
    // survives is the last colour each of them was cleared to. Checking both
    // is what catches an Acquire that hands back the same image every time.
    const std::array<size_t, 2> lastFrameForImage{2u, 1u};
    for (uint32_t index = 0u; index < 2u; index++)
    {
        INFO("image index " << index);

        const std::array<std::byte, 4> expected =
            ExpectedTexel(target->GetFormat(), kFrameColors[lastFrameForImage[index]]);

        const std::vector<std::byte> pixels = RhiTest::ReadRenderedTexture(
            device, imagesByIndex[index], kExtent, target->GetFormat(),
            TextureLayout::ShaderResource, AsOffscreen(*target).TakePendingSignal(index));
        REQUIRE(pixels.size() == static_cast<size_t>(kExtent.Width) * kExtent.Height * 4u);

        // Every texel, not a sample of them: a copy that got the row pitch
        // wrong still produces the right bytes at the origin.
        size_t mismatches = 0u;
        for (size_t texel = 0; texel < pixels.size() / 4u; texel++)
        {
            for (size_t channel = 0; channel < 4u; channel++)
            {
                if (pixels[texel * 4u + channel] != expected[channel])
                    mismatches++;
            }
        }
        CHECK(mismatches == 0u);
    }

    // imagesByIndex only exists to prove the two acquires handed back two
    // different images; the readbacks above name them by index instead.
    CHECK(imagesByIndex[0] != imagesByIndex[1]);
}

TEST_CASE("Recreating an offscreen target resizes it", "[rhi][gpu][present]")
{
    IDevice& device = RhiTest::RequireDevice();
    const RhiTest::ValidationGuard guard(device);

    const std::unique_ptr<IPresentTarget> target =
        device.CreatePresentTarget(PresentTargetDesc{.Extent = kExtent, .FramesInFlight = 2u});

    // A frame first, so the recreate below has a signalled render-complete
    // semaphore and a live image to tear down rather than a pristine target.
    {
        const FrameCommands frame = MakeFrameCommands(device);
        const AcquiredImage acquired = target->Acquire();
        RecordClearFrame(frame.List, acquired, kFrameColors[0], kExtent);
        SubmitFrame(device, *frame.List, acquired.WaitSemaphores,
                    target->GetRenderCompleteSemaphore(acquired.Index));
        CHECK(target->Present(acquired.Index));
        device.WaitIdle();
    }

    constexpr Extent2D kSmaller{64u, 200u};
    REQUIRE(target->Recreate(kSmaller));
    CHECK(target->GetExtent() == kSmaller);
    CHECK(target->GetImageCount() == 2u);

    // Rebuilt from scratch, so the first pass over the new images has nothing
    // outstanding to wait on even though the old ones did.
    const AcquiredImage acquired = target->Acquire();
    CHECK(acquired.WaitSemaphores.empty());

    const FrameCommands frame = MakeFrameCommands(device);
    RecordClearFrame(frame.List, acquired, kFrameColors[1], kSmaller);
    SubmitFrame(device, *frame.List, acquired.WaitSemaphores,
                target->GetRenderCompleteSemaphore(acquired.Index));
    CHECK(target->Present(acquired.Index));

    device.WaitIdle();
}

/**
 * A zero extent is the one request that cannot be met, and the answer is the
 * same "nothing was touched, ask again" a minimised window gets from a
 * swapchain — so a caller that resizes through zero needs no special case for
 * which kind of target it holds.
 */
TEST_CASE("Recreating an offscreen target at a zero extent changes nothing", "[rhi][gpu][present]")
{
    IDevice& device = RhiTest::RequireDevice();
    const RhiTest::ValidationGuard guard(device);

    const std::unique_ptr<IPresentTarget> target =
        device.CreatePresentTarget(PresentTargetDesc{.Extent = kExtent, .FramesInFlight = 2u});

    CHECK_FALSE(target->Recreate(Extent2D{0u, 0u}));
    CHECK_FALSE(target->Recreate(Extent2D{kExtent.Width, 0u}));
    CHECK_FALSE(target->Recreate(Extent2D{0u, kExtent.Height}));

    CHECK(target->GetExtent() == kExtent);

    // Still usable: the images the failed recreates left alone are the ones
    // that were already there.
    const AcquiredImage acquired = target->Acquire();
    CHECK_FALSE(acquired.bNeedsRecreate);
    CHECK(acquired.Texture.IsValid());
}

/**
 * The target owns its images, unlike a swapchain's, so destroying it has to
 * give every one of them back. A leak here would be invisible in a windowed run
 * and would grow with every resize in a headless one.
 */
TEST_CASE("An offscreen target frees its images when it is destroyed", "[rhi][gpu][present]")
{
    IDevice& device = RhiTest::RequireDevice();
    const RhiTest::ValidationGuard guard(device);

    const uint32_t texturesBefore = device.GetLiveTextureCount();
    const uint32_t viewsBefore = device.GetLiveTextureViewCount();

    {
        const std::unique_ptr<IPresentTarget> target =
            device.CreatePresentTarget(PresentTargetDesc{.Extent = kExtent, .FramesInFlight = 3u});

        CHECK(device.GetLiveTextureCount() == texturesBefore + 3u);
        CHECK(device.GetLiveTextureViewCount() == viewsBefore + 3u);

        REQUIRE(target->Recreate(Extent2D{128u, 128u}));

        // A recreate replaces them rather than adding to them, which is the
        // half of a resize that a fixed count cannot tell you about.
        CHECK(device.GetLiveTextureCount() == texturesBefore + 3u);
        CHECK(device.GetLiveTextureViewCount() == viewsBefore + 3u);
    }

    CHECK(device.GetLiveTextureCount() == texturesBefore);
    CHECK(device.GetLiveTextureViewCount() == viewsBefore);
}

/**
 * Step 39's first check: the bytes that come back are the bytes that were
 * rendered, in the target's own channel order.
 *
 * The clear components are 0 or 1 and the four channels differ from one
 * another, so a swizzled readback, a stale image and a half-written one are all
 * distinguishable from a correct result — which a grey or a black clear would
 * not be.
 */
TEST_CASE("Readback returns the exact pixels of a solid clear", "[rhi][gpu][present]")
{
    IDevice& device = RhiTest::RequireDevice();
    const RhiTest::ValidationGuard guard(device);

    const std::unique_ptr<IPresentTarget> target =
        device.CreatePresentTarget(PresentTargetDesc{.Extent = kExtent, .FramesInFlight = 2u});

    constexpr std::array<float, 4> kColor{1.f, 0.f, 1.f, 1.f};

    const FrameCommands frame = MakeFrameCommands(device);
    const AcquiredImage acquired = target->Acquire();
    RecordClearFrame(frame.List, acquired, kColor, kExtent);
    SubmitFrame(device, *frame.List, acquired.WaitSemaphores,
                target->GetRenderCompleteSemaphore(acquired.Index));
    REQUIRE(target->Present(acquired.Index));

    const std::vector<std::byte> pixels =
        RhiTest::ReadRenderedTexture(device, acquired.Texture, target->GetExtent(),
                                     target->GetFormat(), TextureLayout::ShaderResource,
                                     AsOffscreen(*target).TakePendingSignal(acquired.Index));

    const uint32_t bytesPerTexel = BytesPerTexel(target->GetFormat());
    REQUIRE(bytesPerTexel == 4u);
    REQUIRE(pixels.size() == static_cast<size_t>(kExtent.Width) * kExtent.Height * bytesPerTexel);

    const std::array<std::byte, 4> expected = ExpectedTexel(target->GetFormat(), kColor);
    size_t mismatches = 0u;
    for (size_t texel = 0; texel < pixels.size() / 4u; texel++)
    {
        for (size_t channel = 0; channel < 4u; channel++)
        {
            if (pixels[texel * 4u + channel] != expected[channel])
                mismatches++;
        }
    }
    CHECK(mismatches == 0u);
}

/**
 * Step 39's second check, and the one a solid colour cannot make: that the
 * returned bytes are tightly packed and row-major at an extent that is neither
 * square nor a power of two.
 *
 * 253x101 with a 100x50 rectangle in one corner pins down every way this can go
 * wrong. A row pitch rounded up to an alignment shears the rectangle
 * diagonally; width and height transposed puts it in the wrong place and
 * changes its shape; a buffer sized from the wrong extent truncates. All three
 * survive a uniform clear untouched, which is why this case exists next to the
 * one above rather than instead of it.
 */
TEST_CASE("Readback packs a non-square, non-power-of-two extent tightly", "[rhi][gpu][present]")
{
    IDevice& device = RhiTest::RequireDevice();
    const RhiTest::ValidationGuard guard(device);

    const std::unique_ptr<IPresentTarget> target =
        device.CreatePresentTarget(PresentTargetDesc{.Extent = kExtent, .FramesInFlight = 2u});

    constexpr std::array<float, 4> kBackground{1.f, 0.f, 0.f, 1.f};
    constexpr std::array<float, 4> kCorner{0.f, 0.f, 1.f, 1.f};
    constexpr uint32_t kCornerWidth = 100u;
    constexpr uint32_t kCornerHeight = 50u;

    const std::array clears{
        ClearRect{.Color = kBackground, .Area = WholeImage(kExtent)},
        ClearRect{.Color = kCorner,
                  .Area = Rect2D{.Extent = {kCornerWidth, kCornerHeight}}},
    };

    const FrameCommands frame = MakeFrameCommands(device);
    const AcquiredImage acquired = target->Acquire();
    RecordClears(frame.List, acquired, clears);
    SubmitFrame(device, *frame.List, acquired.WaitSemaphores,
                target->GetRenderCompleteSemaphore(acquired.Index));
    REQUIRE(target->Present(acquired.Index));

    const std::vector<std::byte> pixels =
        RhiTest::ReadRenderedTexture(device, acquired.Texture, target->GetExtent(),
                                     target->GetFormat(), TextureLayout::ShaderResource,
                                     AsOffscreen(*target).TakePendingSignal(acquired.Index));

    const uint32_t bytesPerTexel = BytesPerTexel(target->GetFormat());
    REQUIRE(pixels.size() == static_cast<size_t>(kExtent.Width) * kExtent.Height * bytesPerTexel);

    const std::array<std::byte, 4> background = ExpectedTexel(target->GetFormat(), kBackground);
    const std::array<std::byte, 4> corner = ExpectedTexel(target->GetFormat(), kCorner);

    // Indexed as row * Width + column, which is the packing being asserted:
    // reading it back this way and getting the rectangle where it was drawn is
    // the whole claim.
    size_t mismatches = 0u;
    for (uint32_t y = 0u; y < kExtent.Height; y++)
    {
        for (uint32_t x = 0u; x < kExtent.Width; x++)
        {
            const bool bInCorner = x < kCornerWidth && y < kCornerHeight;
            const std::array<std::byte, 4>& expected = bInCorner ? corner : background;

            const size_t texel = static_cast<size_t>(y) * kExtent.Width + x;
            for (size_t channel = 0; channel < 4u; channel++)
            {
                if (pixels[texel * 4u + channel] != expected[channel])
                    mismatches++;
            }
        }
    }
    CHECK(mismatches == 0u);
}

/**
 * The staging buffer a read allocates is freed on the way out, and the copy it
 * fenced on is finished by then. Neither is visible from the bytes returned, so
 * the counters say it instead: a read that leaked its buffer would grow the
 * device's live count once per capture, which in a run capturing every frame is
 * a leak that scales with the run.
 */
TEST_CASE("Readback leaves nothing behind on the device", "[rhi][gpu][present]")
{
    IDevice& device = RhiTest::RequireDevice();
    const RhiTest::ValidationGuard guard(device);

    const std::unique_ptr<IPresentTarget> target =
        device.CreatePresentTarget(PresentTargetDesc{.Extent = kExtent, .FramesInFlight = 2u});

    const uint32_t buffersBefore = device.GetLiveBufferCount();

    for (int capture = 0; capture < 3; capture++)
    {
        const FrameCommands frame = MakeFrameCommands(device);
        const AcquiredImage acquired = target->Acquire();
        RecordClearFrame(frame.List, acquired, kFrameColors[0], kExtent);
        SubmitFrame(device, *frame.List, acquired.WaitSemaphores,
                    target->GetRenderCompleteSemaphore(acquired.Index));
        REQUIRE(target->Present(acquired.Index));

        const std::vector<std::byte> pixels =
            RhiTest::ReadRenderedTexture(device, acquired.Texture, target->GetExtent(),
                                     target->GetFormat(), TextureLayout::ShaderResource,
                                     AsOffscreen(*target).TakePendingSignal(acquired.Index));
        CHECK_FALSE(pixels.empty());
    }

    CHECK(device.GetLiveBufferCount() == buffersBefore);
}
