#include <catch2/catch_test_macros.hpp>

#include <rhi/Diagnostics.h>
#include <rhi/Handles.h>
#include <rhi/IDevice.h>
#include <rhi/UniqueHandle.h>

#include <memory>
#include <utility>
#include <vector>

/**
 * CPU-only: UniqueHandle never touches a driver, it only decides when to call
 * IDevice::Destroy. That decision is the whole of its value — the handle model
 * puts resource release on the caller (plan D2), and this is what puts it back
 * on the compiler — so it is worth testing without needing an ICD present.
 */
using namespace Hikari::Rhi;

namespace
{
/**
 * An IDevice that allocates nothing and records what it was asked to destroy.
 * Every resource kind hands out handles from one index counter and records its
 * destructions in its own vector, so a test can check that UniqueHandle called
 * the right overload as well as that it called one at all.
 */
class RecordingDevice final : public IDevice
{
public:
    const DeviceCaps& GetCaps() const override { return m_Caps; }
    Diagnostics& GetDiagnostics() override { return m_Diagnostics; }
    void WaitIdle() override {}

    BufferHandle CreateBuffer(const BufferDesc&) override
    {
        return BufferHandle::FromIndexAndGeneration(m_NextIndex++, 0u);
    }

    void Destroy(BufferHandle handle) override { Destroyed.push_back(handle); }

    void* GetMappedData(BufferHandle) override { return nullptr; }

    uint32_t GetLiveBufferCount() const override
    {
        return m_NextIndex - static_cast<uint32_t>(Destroyed.size());
    }

    TextureHandle CreateTexture(const TextureDesc&) override
    {
        return TextureHandle::FromIndexAndGeneration(m_NextIndex++, 0u);
    }

    void Destroy(TextureHandle handle) override { DestroyedTextures.push_back(handle); }

    TextureViewHandle CreateTextureView(const TextureViewDesc&) override
    {
        return TextureViewHandle::FromIndexAndGeneration(m_NextIndex++, 0u);
    }

    void Destroy(TextureViewHandle handle) override { DestroyedViews.push_back(handle); }

    SamplerHandle CreateSampler(const SamplerDesc&) override
    {
        return SamplerHandle::FromIndexAndGeneration(m_NextIndex++, 0u);
    }

    void Destroy(SamplerHandle handle) override { DestroyedSamplers.push_back(handle); }

    const TextureDesc* GetTextureDesc(TextureHandle) const override { return nullptr; }

    std::unique_ptr<IUploadContext> CreateUploadContext(const UploadContextDesc&) override
    {
        return nullptr;
    }

    std::unique_ptr<ICommandAllocator> CreateCommandAllocator(const CommandAllocatorDesc&) override
    {
        return nullptr;
    }

    FenceHandle CreateFence(const FenceDesc&) override
    {
        return FenceHandle::FromIndexAndGeneration(m_NextIndex++, 0u);
    }

    void Destroy(FenceHandle handle) override { DestroyedFences.push_back(handle); }

    uint32_t GetLiveFenceCount() const override
    {
        return static_cast<uint32_t>(DestroyedFences.size());
    }

    void WaitForFence(FenceHandle, uint64_t) override {}

    void Submit(const SubmitDesc&) override {}

    std::unique_ptr<IPipelineCache> CreatePipelineCache(const PipelineCacheDesc&) override
    {
        return nullptr;
    }

    std::unique_ptr<IPresentTarget> CreatePresentTarget(const PresentTargetDesc&) override
    {
        return nullptr;
    }

    uint32_t GetLiveTextureCount() const override
    {
        return static_cast<uint32_t>(DestroyedTextures.size());
    }

    uint32_t GetLiveTextureViewCount() const override
    {
        return static_cast<uint32_t>(DestroyedViews.size());
    }

    uint32_t GetLiveSamplerCount() const override
    {
        return static_cast<uint32_t>(DestroyedSamplers.size());
    }

    std::vector<BufferHandle> Destroyed;
    std::vector<TextureHandle> DestroyedTextures;
    std::vector<TextureViewHandle> DestroyedViews;
    std::vector<SamplerHandle> DestroyedSamplers;
    std::vector<FenceHandle> DestroyedFences;

private:
    DeviceCaps m_Caps{};
    Diagnostics m_Diagnostics{Diagnostics::Desc{}};
    uint32_t m_NextIndex = 0u;
};
} // namespace

TEST_CASE("A UniqueHandle destroys what it holds when it goes out of scope", "[RhiUniqueHandle]")
{
    RecordingDevice device;
    const BufferHandle handle = device.CreateBuffer(BufferDesc{});

    {
        UniqueHandle<BufferHandle> owned(device, handle);
        REQUIRE(owned.Get() == handle);
        REQUIRE(owned.IsValid());
        REQUIRE(device.Destroyed.empty());
    }

    REQUIRE(device.Destroyed.size() == 1u);
    REQUIRE(device.Destroyed.front() == handle);
}

TEST_CASE("An empty UniqueHandle destroys nothing", "[RhiUniqueHandle]")
{
    RecordingDevice device;

    {
        UniqueHandle<BufferHandle> empty;
        REQUIRE_FALSE(empty.IsValid());
    }

    REQUIRE(device.Destroyed.empty());
}

TEST_CASE("Moving a UniqueHandle transfers the resource rather than sharing it",
          "[RhiUniqueHandle]")
{
    RecordingDevice device;
    const BufferHandle handle = device.CreateBuffer(BufferDesc{});

    {
        UniqueHandle<BufferHandle> source(device, handle);
        UniqueHandle<BufferHandle> destination(std::move(source));

        REQUIRE(destination.Get() == handle);

        // The moved-from wrapper must not also destroy it: a double release is
        // what the generation counter would otherwise have to catch at runtime.
        REQUIRE_FALSE(source.IsValid()); // NOLINT(bugprone-use-after-move)
    }

    REQUIRE(device.Destroyed.size() == 1u);
}

TEST_CASE("Move-assigning destroys what the target was already holding", "[RhiUniqueHandle]")
{
    RecordingDevice device;
    const BufferHandle replaced = device.CreateBuffer(BufferDesc{});
    const BufferHandle survivor = device.CreateBuffer(BufferDesc{});

    {
        UniqueHandle<BufferHandle> target(device, replaced);
        UniqueHandle<BufferHandle> source(device, survivor);

        target = std::move(source);

        // The overwritten resource goes now, not at end of scope — otherwise
        // reassigning a member in a resize path would leak until shutdown.
        REQUIRE(device.Destroyed.size() == 1u);
        REQUIRE(device.Destroyed.front() == replaced);
    }

    REQUIRE(device.Destroyed.size() == 2u);
    REQUIRE(device.Destroyed.back() == survivor);
}

TEST_CASE("Release hands ownership out without destroying", "[RhiUniqueHandle]")
{
    RecordingDevice device;
    const BufferHandle handle = device.CreateBuffer(BufferDesc{});

    {
        UniqueHandle<BufferHandle> owned(device, handle);
        const BufferHandle released = owned.Release();

        REQUIRE(released == handle);
        REQUIRE_FALSE(owned.IsValid());
    }

    REQUIRE(device.Destroyed.empty());
}

TEST_CASE("Reset destroys immediately and leaves the wrapper empty", "[RhiUniqueHandle]")
{
    RecordingDevice device;
    const BufferHandle handle = device.CreateBuffer(BufferDesc{});

    UniqueHandle<BufferHandle> owned(device, handle);
    owned.Reset();

    REQUIRE(device.Destroyed.size() == 1u);
    REQUIRE_FALSE(owned.IsValid());

    // A second Reset must not destroy the handle again.
    owned.Reset();
    REQUIRE(device.Destroyed.size() == 1u);
}
