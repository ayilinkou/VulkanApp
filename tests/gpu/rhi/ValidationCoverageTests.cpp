#include <catch2/catch_test_macros.hpp>

#include <memory>

#include <rhi/ICommandAllocator.h>
#include <rhi/ICommandList.h>
#include <rhi/IDevice.h>
#include <rhi/Submit.h>
#include <rhi/UniqueHandle.h>

#include "RhiTestFixture.h"

/**
 * What the validation setup can actually catch, as opposed to what it is
 * configured to catch.
 *
 * Every other gpu case asserts that validation stayed quiet. That is worth
 * nothing unless something proves validation would have spoken, and the
 * difference is not academic: `backlog.md` carried a row for a while claiming
 * the suite could not detect a missing cross-submit dependency, on the strength
 * of an offscreen read that kept passing when its wait semaphore was removed.
 * It could detect one all along -- the read had no hazard to find, because a
 * barrier and submission order were already supplying the dependency the
 * semaphore was being credited with.
 *
 * A case here deliberately commits the error and requires that it was reported,
 * then clears the counters so the deliberate error does not fail the run. It is
 * the positive control for every negative assertion elsewhere.
 */
using namespace Hikari::Rhi;

/**
 * A read-after-write split across two submissions to one queue, with no
 * barrier, no semaphore and no fence between them.
 *
 * Buffers rather than textures on purpose. An image carries layout transitions,
 * and a transition is itself ordered against other transitions on the same
 * queue, so an image version of this test could pass for a reason that has
 * nothing to do with the hazard being detected. A buffer has no such implicit
 * ordering: if this is reported, synchronization validation is genuinely
 * tracking hazards across submissions.
 */
TEST_CASE("Synchronization validation detects a cross-submit hazard", "[rhi][gpu][validation]")
{
    IDevice& device = RhiTest::RequireDevice();
    Diagnostics& diagnostics = device.GetDiagnostics();

    // Not ValidationGuard: this case wants errors rather than their absence, so
    // it manages the counters itself and clears them on the way out.
    diagnostics.Reset();

    constexpr uint64_t kSize = 256u;

    const UniqueHandle<BufferHandle> source(
        device, device.CreateBuffer(BufferDesc{.Size = kSize,
                                               .Usage = BufferUsage::CopySrc,
                                               .Access = MemoryAccess::CpuToGpu,
                                               .DebugName = "Hazard Source"}));
    const UniqueHandle<BufferHandle> middle(
        device, device.CreateBuffer(BufferDesc{.Size = kSize,
                                               .Usage = BufferUsage::CopySrc | BufferUsage::CopyDst,
                                               .Access = MemoryAccess::GpuOnly,
                                               .DebugName = "Hazard Middle"}));
    const UniqueHandle<BufferHandle> destination(
        device, device.CreateBuffer(BufferDesc{.Size = kSize,
                                               .Usage = BufferUsage::CopyDst,
                                               .Access = MemoryAccess::GpuToCpu,
                                               .DebugName = "Hazard Destination"}));

    // One allocator each, because two lists recording at once may not share one
    // (ICommandAllocator).
    const std::unique_ptr<ICommandAllocator> writeAllocator = device.CreateCommandAllocator(
        CommandAllocatorDesc{.Queue = QueueType::Graphics, .DebugName = "Hazard Write"});
    const std::unique_ptr<ICommandAllocator> readAllocator = device.CreateCommandAllocator(
        CommandAllocatorDesc{.Queue = QueueType::Graphics, .DebugName = "Hazard Read"});

    ICommandList& writeList = writeAllocator->Acquire();
    writeList.Begin();
    writeList.CopyBuffer(source.Get(), middle.Get(), BufferCopyRegion{.Size = kSize});
    writeList.End();

    ICommandList& readList = readAllocator->Acquire();
    readList.Begin();
    readList.CopyBuffer(middle.Get(), destination.Get(), BufferCopyRegion{.Size = kSize});
    readList.End();

    // Two submissions to one queue with nothing ordering them: no fence between,
    // no semaphore, and no barrier inside either list.
    ICommandList* pWrite = &writeList;
    ICommandList* pRead = &readList;

    const UniqueHandle<FenceHandle> fence(
        device, device.CreateFence(FenceDesc{.DebugName = "Hazard Fence"}));
    constexpr uint64_t kDone = 1u;
    const FenceOperation signal{.Fence = fence.Get(), .Value = kDone};

    device.Submit(SubmitDesc{.Queue = QueueType::Graphics, .CommandLists = {&pWrite, 1u}});
    device.Submit(SubmitDesc{.Queue = QueueType::Graphics,
                             .CommandLists = {&pRead, 1u},
                             .SignalFences = {&signal, 1u}});

    device.WaitForFence(fence.Get(), kDone);

    // The whole point. If this ever stops holding, every "zero validation
    // errors" assertion in the gpu suite has quietly stopped meaning anything,
    // and this is the case that says so.
    CHECK(diagnostics.ErrorCount() >= 1u);

    // The hazard was deliberate, so the errors it produced are not a result. The
    // device outlives this case, and the next case's ValidationGuard resets on
    // construction, but clearing here keeps a failure elsewhere from inheriting
    // this one's messages.
    diagnostics.Reset();
}
