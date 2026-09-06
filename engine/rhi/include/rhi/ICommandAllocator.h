#pragma once

#include <string>

#include <rhi/RhiTypes.h>

namespace Hikari::Rhi
{
class ICommandList;

/**
 * What a command allocator is created for.
 *
 * The queue type is fixed at creation because both backends fix it there: a
 * VkCommandPool carries a queue family index and an ID3D12CommandAllocator
 * carries a D3D12_COMMAND_LIST_TYPE. Lists recorded from an allocator can only
 * be submitted to a queue of that kind.
 */
struct CommandAllocatorDesc
{
    QueueType Queue = QueueType::Graphics;

    std::string DebugName;
};

/**
 * The storage command lists record into, and the unit that storage is recycled
 * in.
 *
 * Caller-owned, deliberately. Both APIs require external synchronization on
 * this object -- vkResetCommandPool and vkAllocateCommandBuffers say so of a
 * VkCommandPool, and D3D12 says the same of an allocator -- so a frame that
 * records on several threads at once needs one per thread. Keeping that in the
 * caller's hands is the point: a thread-local pool hidden inside the RHI would
 * turn a rule the caller can see and satisfy into one it cannot. The
 * arrangement this engine uses is one allocator per recorder per frame in
 * flight, which is what makes its parallel recording safe -- two threads never
 * touch one allocator.
 */
class ICommandAllocator
{
public:
    virtual ~ICommandAllocator() = default;

    ICommandAllocator(const ICommandAllocator&) = delete;
    ICommandAllocator& operator=(const ICommandAllocator&) = delete;
    ICommandAllocator(ICommandAllocator&&) = delete;
    ICommandAllocator& operator=(ICommandAllocator&&) = delete;

    /**
     * A list to record into, ready for Begin(). Valid until the next Reset().
     *
     * At most one of an allocator's lists may be recording -- between Begin()
     * and End() -- at a time. Vulkan permits several; D3D12 permits one, so the
     * neutral rule is D3D12's. Acquire a second list only once the first has
     * been ended.
     *
     * The list belongs to the allocator, which says nothing about how a backend
     * stores it: a VkCommandBuffer is owned by its pool, while a D3D12 command
     * list is a device object re-pointed at an allocator by Reset. All a caller
     * may rely on is the sentence below.
     */
    virtual ICommandList& Acquire() = 0;

    /**
     * Recycles the allocator, invalidating every list Acquire() has returned.
     *
     * Must not be called while work recorded into those lists may still be
     * executing. Neither API tracks that for you, and neither will tell you
     * afterwards -- the frame loop's fence wait is what makes it safe here.
     * This is the one operation on the type that can be wrong in a way nothing
     * catches, which is why the allocator is an object the caller holds rather
     * than an argument to a device method.
     */
    virtual void Reset() = 0;

protected:
    ICommandAllocator() = default;
};
} // namespace Hikari::Rhi
