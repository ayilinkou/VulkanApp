#pragma once

#include <cstdint>
#include <span>
#include <string>

#include <rhi/Handles.h>
#include <rhi/RhiTypes.h>

namespace Hikari::Rhi
{
class ICommandList;

/**
 * A fence, and the value it is waited on or signalled to reach.
 *
 * Fences are monotonic counters rather than the signalled/unsignalled flags
 * Vulkan's VkFence uses, because that is the only primitive D3D12 has: an
 * ID3D12Fence and a value. Vulkan's equivalent is a timeline semaphore, core
 * since 1.2. Modelling the intersection means a wait can name a point in the
 * past and return immediately, which is what makes "wait for the frame that
 * used this slot" expressible without resetting anything (plan D5).
 */
struct FenceOperation
{
    FenceHandle Fence{};
    uint64_t Value = 0u;
};

struct FenceDesc
{
    /** The counter's starting point. Waits for this value or lower return at once. */
    uint64_t InitialValue = 0u;

    std::string DebugName;
};

/**
 * One submission to one queue.
 *
 * Lists execute in the order given. Every list must have been ended, and every
 * one must have come from an allocator created for this queue type.
 *
 * The two semaphore spans exist only for present targets. Nothing else produces
 * a SemaphoreHandle, and nothing should: a swapchain image is acquired and
 * presented with binary semaphores because Vulkan requires it
 * (VUID-vkAcquireNextImageKHR-semaphore-03265 and
 * VUID-vkQueuePresentKHR-pWaitSemaphores-03267 both demand
 * VK_SEMAPHORE_TYPE_BINARY), and D3D12's swap chain has no equivalent object at
 * all. They carry no stage: Vulkan wants one, D3D12 has none, and the only
 * semaphores that reach here guard writes to an acquired image, so the backend
 * picks the first stage that could write one rather than asking a caller to
 * name a stage it cannot express portably.
 */
struct SubmitDesc
{
    QueueType Queue = QueueType::Graphics;

    std::span<ICommandList* const> CommandLists{};

    std::span<const FenceOperation> WaitFences{};
    std::span<const FenceOperation> SignalFences{};

    std::span<const SemaphoreHandle> WaitSemaphores{};
    std::span<const SemaphoreHandle> SignalSemaphores{};
};
} // namespace Hikari::Rhi
