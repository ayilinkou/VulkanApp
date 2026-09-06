#include <catch2/catch_test_macros.hpp>

#include <rhi/Barrier.h>
#include <rhi/BarrierPresets.h>
#include <rhi/BufferDesc.h>
#include <rhi/Diagnostics.h>
#include <rhi/RhiTypes.h>
#include <rhi/SamplerDesc.h>
#include <rhi/TextureDesc.h>
#include <rhi/TextureViewDesc.h>

#include "vulkan/VulkanConversions.h"

#include <set>
#include <stdexcept>


/**
 * CPU-only: no instance, no device, no ICD needed. These are table tests, and
 * the tables are the part of the RHI most likely to be wrong in a way that
 * compiles — a mismapped format or a barrier missing a stage bit produces
 * corruption or an intermittent hazard, not a build failure.
 */
using namespace Hikari::Core;
using namespace Hikari::Rhi;
using namespace Hikari::Rhi::Vulkan;

namespace
{
/**
 * Checks a two-way mapping: every neutral value survives ToVk followed by
 * FromVk, and no two neutral values map to the same Vulkan value.
 *
 * The second half is not just tidiness. FromVk is derived by searching for the
 * neutral value whose ToVk matches, so if two neutral values shared a Vulkan
 * value, FromVk would return whichever happened to come first in the array and
 * the round-trip would quietly fail for the other one.
 */
template <typename NeutralEnum, size_t N>
void RequireRoundTrips(const std::array<NeutralEnum, N>& all)
{
    std::set<uint64_t> seen;

    for (const NeutralEnum value : all)
    {
        const auto converted = ToVk(value);

        REQUIRE(FromVk(converted) == value);

        // No two neutral values may share a Vulkan value.
        const auto asInteger = static_cast<uint64_t>(converted);
        REQUIRE_FALSE(seen.contains(asInteger));
        seen.insert(asInteger);
    }

    REQUIRE(seen.size() == all.size());
}
} // namespace

TEST_CASE("Every Format round-trips and no two Formats share a VkFormat", "[RhiConversions]")
{
    RequireRoundTrips(kAllFormats);
}

TEST_CASE("Formats map to the VkFormat their name claims", "[RhiConversions]")
{
    // Spot checks against the Vulkan header, because a round-trip proves the
    // table is self-consistent, not that it is right. These are the mappings a
    // transposition would silently survive: swapping RGBA8Unorm and RGBA8Srgb
    // still round-trips, and only shows up as everything being too dark or too
    // bright.
    REQUIRE(ToVk(Format::Undefined) == vk::Format::eUndefined);
    REQUIRE(ToVk(Format::R8Unorm) == vk::Format::eR8Unorm);
    REQUIRE(ToVk(Format::RGBA8Unorm) == vk::Format::eR8G8B8A8Unorm);
    REQUIRE(ToVk(Format::RGBA8Srgb) == vk::Format::eR8G8B8A8Srgb);
    REQUIRE(ToVk(Format::BGRA8Unorm) == vk::Format::eB8G8R8A8Unorm);
    REQUIRE(ToVk(Format::RGBA16Float) == vk::Format::eR16G16B16A16Sfloat);

    // The vertex-attribute formats, where a transposition is just as quiet:
    // swapping RG32Float and RGB32Float still round-trips and only shows up as
    // geometry reading one component of garbage.
    REQUIRE(ToVk(Format::RG32Float) == vk::Format::eR32G32Sfloat);
    REQUIRE(ToVk(Format::RGB32Float) == vk::Format::eR32G32B32Sfloat);
    REQUIRE(ToVk(Format::RGBA32Float) == vk::Format::eR32G32B32A32Sfloat);

    REQUIRE(ToVk(Format::D16Unorm) == vk::Format::eD16Unorm);
    REQUIRE(ToVk(Format::D32Float) == vk::Format::eD32Sfloat);
    REQUIRE(ToVk(Format::D24UnormS8Uint) == vk::Format::eD24UnormS8Uint);
    REQUIRE(ToVk(Format::D32FloatS8Uint) == vk::Format::eD32SfloatS8Uint);
}

TEST_CASE("A VkFormat outside the curated set is rejected rather than guessed at",
          "[RhiConversions]")
{
    // Deliberately a real format the enum does not carry. Three-component 8-bit
    // is the safest specimen available: it is poorly supported in hardware and
    // has no use here, so it is unlikely to be curated in later and quietly turn
    // this case into an assertion about nothing.
    //
    // This used to name a vertex-attribute format, which stopped being outside
    // the set when pipeline creation was neutralized and RG/RGB/RGBA32Float were
    // added -- the specimen expired, not the rule. Silently returning Undefined
    // here would still turn a missing mapping into a wrong-format resource much
    // later, which is what this guards.
    REQUIRE_THROWS_AS(FromVk(vk::Format::eR8G8B8Unorm), std::runtime_error);
}

TEST_CASE("Depth and stencil predicates agree with the format list", "[RhiConversions]")
{
    REQUIRE_FALSE(IsDepthFormat(Format::Undefined));
    REQUIRE_FALSE(IsDepthFormat(Format::RGBA8Unorm));
    REQUIRE(IsDepthFormat(Format::D16Unorm));
    REQUIRE(IsDepthFormat(Format::D32Float));
    REQUIRE(IsDepthFormat(Format::D24UnormS8Uint));
    REQUIRE(IsDepthFormat(Format::D32FloatS8Uint));

    REQUIRE_FALSE(HasStencilComponent(Format::D16Unorm));
    REQUIRE_FALSE(HasStencilComponent(Format::D32Float));
    REQUIRE(HasStencilComponent(Format::D24UnormS8Uint));
    REQUIRE(HasStencilComponent(Format::D32FloatS8Uint));

    // A stencil component implies a depth one for every format in the set, so
    // nothing may claim stencil without depth.
    for (const Format format : kAllFormats)
    {
        if (HasStencilComponent(format))
            REQUIRE(IsDepthFormat(format));
    }
}

TEST_CASE("DefaultAspect follows the format's components", "[RhiConversions]")
{
    REQUIRE(DefaultAspect(Format::RGBA8Unorm) == TextureAspect::Color);
    REQUIRE(DefaultAspect(Format::Undefined) == TextureAspect::Color);
    REQUIRE(DefaultAspect(Format::D32Float) == TextureAspect::Depth);
    REQUIRE(DefaultAspect(Format::D24UnormS8Uint) ==
            (TextureAspect::Depth | TextureAspect::Stencil));

    REQUIRE(ToVk(DefaultAspect(Format::RGBA8Unorm)) == vk::ImageAspectFlagBits::eColor);
    REQUIRE(ToVk(DefaultAspect(Format::D32Float)) == vk::ImageAspectFlagBits::eDepth);
    REQUIRE(ToVk(DefaultAspect(Format::D32FloatS8Uint)) ==
            (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil));
}

TEST_CASE("Sampler and texture scalar enums map both ways", "[RhiConversions]")
{
    RequireRoundTrips(kAllSampleCounts);
    RequireRoundTrips(kAllTextureDimensions);
    RequireRoundTrips(kAllTextureViewDimensions);
    RequireRoundTrips(kAllFilters);
    RequireRoundTrips(kAllMipmapModes);
    RequireRoundTrips(kAllAddressModes);
    RequireRoundTrips(kAllCompareOps);
    RequireRoundTrips(kAllBorderColors);
}

TEST_CASE("Every PresentMode round-trips", "[RhiConversions]")
{
    // The report names the mode a run actually presented in, so a mapping that
    // lost a mode would mislabel the measurement rather than fail loudly.
    RequireRoundTrips(kAllPresentModes);
}

TEST_CASE("SampleCount values are the sample counts themselves", "[RhiConversions]")
{
    // The enum's numeric values are load-bearing: they are the sample counts,
    // which is what lets a count be turned into a Vulkan bit without a table.
    REQUIRE(static_cast<uint32_t>(SampleCount::X1) == 1u);
    REQUIRE(static_cast<uint32_t>(SampleCount::X4) == 4u);
    REQUIRE(ToVk(SampleCount::X1) == vk::SampleCountFlagBits::e1);
    REQUIRE(ToVk(SampleCount::X4) == vk::SampleCountFlagBits::e4);
    REQUIRE(ToVk(SampleCount::X16) == vk::SampleCountFlagBits::e16);
}

TEST_CASE("A queue role is served by any family capable of it", "[RhiConversions]")
{
    // The masks real drivers report, named for what they are.
    constexpr vk::QueueFlags universal = vk::QueueFlagBits::eGraphics |
                                         vk::QueueFlagBits::eCompute |
                                         vk::QueueFlagBits::eTransfer;
    constexpr vk::QueueFlags asyncCompute =
        vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer;
    constexpr vk::QueueFlags dedicatedTransfer = vk::QueueFlagBits::eTransfer;

    // One family may serve every role, and normally does — this application
    // submits graphics, compute and copies to a single queue today.
    REQUIRE(FamilySupports(universal, QueueType::Graphics));
    REQUIRE(FamilySupports(universal, QueueType::Compute));
    REQUIRE(FamilySupports(universal, QueueType::Copy));

    REQUIRE(FamilySupports(asyncCompute, QueueType::Compute));
    REQUIRE(FamilySupports(asyncCompute, QueueType::Copy));
    REQUIRE_FALSE(FamilySupports(asyncCompute, QueueType::Graphics));

    // A dedicated transfer family serves Copy and nothing else, which is the
    // arrangement uploads are routed onto when a device has one.
    REQUIRE(FamilySupports(dedicatedTransfer, QueueType::Copy));
    REQUIRE_FALSE(FamilySupports(dedicatedTransfer, QueueType::Graphics));
    REQUIRE_FALSE(FamilySupports(dedicatedTransfer, QueueType::Compute));
}

TEST_CASE("A family that can copy without advertising the transfer bit still serves Copy",
          "[RhiConversions]")
{
    // The spec makes reporting VK_QUEUE_TRANSFER_BIT *optional* for a family
    // that already advertises graphics or compute, since such a family can
    // always perform transfers. This is the case a naive `flags & eTransfer`
    // test gets wrong: the family copies perfectly well but never says so, and
    // the copy work has nowhere to go.
    constexpr vk::QueueFlags graphicsOnly = vk::QueueFlagBits::eGraphics;
    constexpr vk::QueueFlags computeOnly = vk::QueueFlagBits::eCompute;

    REQUIRE((graphicsOnly & vk::QueueFlagBits::eTransfer) == vk::QueueFlags{});
    REQUIRE(FamilySupports(graphicsOnly, QueueType::Copy));

    REQUIRE((computeOnly & vk::QueueFlagBits::eTransfer) == vk::QueueFlags{});
    REQUIRE(FamilySupports(computeOnly, QueueType::Copy));
}

TEST_CASE("A family advertising nothing serves no role", "[RhiConversions]")
{
    for (const QueueType role : kAllQueueTypes)
        REQUIRE_FALSE(FamilySupports(vk::QueueFlags{}, role));
}

TEST_CASE("MemoryAccess round-trips through its VMA usage and flags", "[RhiConversions]")
{
    for (const MemoryAccess access : kAllMemoryAccesses)
        REQUIRE(FromVk(ToVk(access)) == access);

    // GpuOnly must not ask for host access: requesting a mapping is what would
    // push VMA away from device-local memory.
    const VmaMemoryParams gpuOnly = ToVk(MemoryAccess::GpuOnly);
    REQUIRE(gpuOnly.Flags == 0u);

    // The two host-visible kinds differ, and specifically in the access-pattern
    // hint — reading back through a sequential-write (potentially
    // write-combined) mapping is legal but pathologically slow.
    const VmaMemoryParams upload = ToVk(MemoryAccess::CpuToGpu);
    const VmaMemoryParams readback = ToVk(MemoryAccess::GpuToCpu);
    REQUIRE(upload != readback);

    REQUIRE((upload.Flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) != 0u);
    REQUIRE((readback.Flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) != 0u);
    REQUIRE((upload.Flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0u);
    REQUIRE((readback.Flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0u);
}

TEST_CASE("An unrecognised VMA parameter pair is rejected", "[RhiConversions]")
{
    const VmaMemoryParams nonsense{.Usage = VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED, .Flags = 0};
    REQUIRE_THROWS_AS(FromVk(nonsense), std::runtime_error);
}

TEST_CASE("Every TextureLayout has a distinct Vulkan layout, except the documented collision",
          "[RhiConversions]")
{
    REQUIRE(ToVk(TextureLayout::Undefined) == vk::ImageLayout::eUndefined);
    REQUIRE(ToVk(TextureLayout::RenderTarget) == vk::ImageLayout::eColorAttachmentOptimal);
    REQUIRE(ToVk(TextureLayout::ShaderResource) == vk::ImageLayout::eShaderReadOnlyOptimal);
    REQUIRE(ToVk(TextureLayout::DepthStencilWrite) ==
            vk::ImageLayout::eDepthStencilAttachmentOptimal);
    REQUIRE(ToVk(TextureLayout::DepthStencilRead) ==
            vk::ImageLayout::eDepthStencilReadOnlyOptimal);
    REQUIRE(ToVk(TextureLayout::CopySrc) == vk::ImageLayout::eTransferSrcOptimal);
    REQUIRE(ToVk(TextureLayout::CopyDst) == vk::ImageLayout::eTransferDstOptimal);
    REQUIRE(ToVk(TextureLayout::Present) == vk::ImageLayout::ePresentSrcKHR);

    // Pinning the collision rather than working around it: Vulkan has no
    // separate "usable from any queue" layout, so Common and UnorderedAccess
    // both become eGeneral. This is why TextureLayout has no FromVk.
    REQUIRE(ToVk(TextureLayout::Common) == vk::ImageLayout::eGeneral);
    REQUIRE(ToVk(TextureLayout::UnorderedAccess) == vk::ImageLayout::eGeneral);

    // Walking the array catches a new layout added without a mapping, and
    // asserts that Common/UnorderedAccess is the *only* pair sharing a Vulkan
    // value — so a future collision has to be justified here rather than
    // silently making two layouts interchangeable.
    std::multiset<vk::ImageLayout> mapped;
    for (const TextureLayout layout : kAllTextureLayouts)
        mapped.insert(ToVk(layout));

    REQUIRE(mapped.size() == kAllTextureLayouts.size());
    REQUIRE(mapped.count(vk::ImageLayout::eGeneral) == 2);

    for (const vk::ImageLayout layout : mapped)
    {
        if (layout != vk::ImageLayout::eGeneral)
            REQUIRE(mapped.count(layout) == 1);
    }
}

TEST_CASE("Every PipelineStage bit maps to a non-empty Vulkan stage mask", "[RhiConversions]")
{
    for (const PipelineStage stage : kAllPipelineStages)
        REQUIRE(ToVk(stage) != vk::PipelineStageFlags2{});

    REQUIRE(ToVk(PipelineStage::None) == vk::PipelineStageFlagBits2::eNone);
    REQUIRE(ToVk(PipelineStage::VertexStage) == vk::PipelineStageFlagBits2::eVertexShader);
    REQUIRE(ToVk(PipelineStage::PixelStage) == vk::PipelineStageFlagBits2::eFragmentShader);
    REQUIRE(ToVk(PipelineStage::ComputeStage) == vk::PipelineStageFlagBits2::eComputeShader);
    REQUIRE(ToVk(PipelineStage::RenderTarget) ==
            vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    REQUIRE(ToVk(PipelineStage::Copy) == vk::PipelineStageFlagBits2::eCopy);
    REQUIRE(ToVk(PipelineStage::All) == vk::PipelineStageFlagBits2::eAllCommands);
}

TEST_CASE("The pixel stage and the render target are different stages", "[RhiConversions]")
{
    // Adjacent in the pipeline but not interchangeable: the fragment shader
    // produces a colour, the colour attachment output writes it. Conflating
    // them under-synchronizes, so the distinction is pinned here.
    REQUIRE(ToVk(PipelineStage::PixelStage) != ToVk(PipelineStage::RenderTarget));
    REQUIRE(ToVk(PipelineStage::PixelStage) == vk::PipelineStageFlagBits2::eFragmentShader);
    REQUIRE(ToVk(PipelineStage::RenderTarget) ==
            vk::PipelineStageFlagBits2::eColorAttachmentOutput);

    // Likewise the vertex stage sits at the opposite end of the pipeline.
    REQUIRE(ToVk(PipelineStage::VertexStage) != ToVk(PipelineStage::RenderTarget));
}

TEST_CASE("DepthStencil covers both fragment-test stages", "[RhiConversions]")
{
    // Vulkan has two depth/stencil test stages because the test can run before
    // the fragment shader (the fast path, rejecting hidden fragments before
    // shading them) or after it, when the shader is what determines the
    // fragment's depth or whether it survives at all. Which applies depends on
    // the pipeline, not on the barrier, so a barrier must name both. Naming one
    // is the classic "renders correctly on this driver" bug, so the pairing is
    // asserted rather than left to the reader of the table.
    const vk::PipelineStageFlags2 stages = ToVk(PipelineStage::DepthStencil);

    REQUIRE((stages & vk::PipelineStageFlagBits2::eEarlyFragmentTests) ==
            vk::PipelineStageFlagBits2::eEarlyFragmentTests);
    REQUIRE((stages & vk::PipelineStageFlagBits2::eLateFragmentTests) ==
            vk::PipelineStageFlagBits2::eLateFragmentTests);

    // The fragment shader sits between the two, and is a separate stage: it is
    // where a texture is sampled, not where depth is tested.
    REQUIRE((stages & vk::PipelineStageFlagBits2::eFragmentShader) ==
            vk::PipelineStageFlags2{});
}

TEST_CASE("Combining PipelineStage bits ORs their Vulkan equivalents", "[RhiConversions]")
{
    const PipelineStage combined = PipelineStage::Copy | PipelineStage::ComputeStage;

    REQUIRE(ToVk(combined) ==
            (vk::PipelineStageFlagBits2::eCopy | vk::PipelineStageFlagBits2::eComputeShader));

    // Order of combination must not matter.
    REQUIRE(ToVk(PipelineStage::ComputeStage | PipelineStage::Copy) == ToVk(combined));
}

TEST_CASE("Every AccessFlags bit maps to a non-empty Vulkan access mask", "[RhiConversions]")
{
    for (const AccessFlags access : kAllAccessFlags)
        REQUIRE(ToVk(access) != vk::AccessFlags2{});

    REQUIRE(ToVk(AccessFlags::None) == vk::AccessFlagBits2::eNone);
    REQUIRE(ToVk(AccessFlags::VertexBufferRead) == vk::AccessFlagBits2::eVertexAttributeRead);
    REQUIRE(ToVk(AccessFlags::IndexBufferRead) == vk::AccessFlagBits2::eIndexRead);
    REQUIRE(ToVk(AccessFlags::ConstantBufferRead) == vk::AccessFlagBits2::eUniformRead);
    REQUIRE(ToVk(AccessFlags::RenderTargetRead) == vk::AccessFlagBits2::eColorAttachmentRead);
    REQUIRE(ToVk(AccessFlags::RenderTargetWrite) == vk::AccessFlagBits2::eColorAttachmentWrite);
    REQUIRE(ToVk(AccessFlags::DepthStencilWrite) ==
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
    REQUIRE(ToVk(AccessFlags::CopySrc) == vk::AccessFlagBits2::eTransferRead);
    REQUIRE(ToVk(AccessFlags::CopyDst) == vk::AccessFlagBits2::eTransferWrite);
}

TEST_CASE("UnorderedAccess widens to both storage read and write", "[RhiConversions]")
{
    // D3D12 treats unordered access as one concept, so the neutral bit covers
    // both directions. Widening a barrier is safe; the test records that the
    // widening is deliberate rather than an oversight in the table.
    const vk::AccessFlags2 access = ToVk(AccessFlags::UnorderedAccess);

    REQUIRE((access & vk::AccessFlagBits2::eShaderStorageRead) ==
            vk::AccessFlagBits2::eShaderStorageRead);
    REQUIRE((access & vk::AccessFlagBits2::eShaderStorageWrite) ==
            vk::AccessFlagBits2::eShaderStorageWrite);
}

TEST_CASE("Buffer and texture usage bits map to their Vulkan counterparts", "[RhiConversions]")
{
    REQUIRE(ToVk(BufferUsage::None) == vk::BufferUsageFlags{});
    REQUIRE(ToVk(BufferUsage::Vertex) == vk::BufferUsageFlagBits::eVertexBuffer);
    REQUIRE(ToVk(BufferUsage::Index) == vk::BufferUsageFlagBits::eIndexBuffer);
    REQUIRE(ToVk(BufferUsage::Uniform) == vk::BufferUsageFlagBits::eUniformBuffer);
    REQUIRE(ToVk(BufferUsage::Storage) == vk::BufferUsageFlagBits::eStorageBuffer);
    REQUIRE(ToVk(BufferUsage::CopySrc) == vk::BufferUsageFlagBits::eTransferSrc);
    REQUIRE(ToVk(BufferUsage::CopyDst) == vk::BufferUsageFlagBits::eTransferDst);

    REQUIRE(ToVk(BufferUsage::Vertex | BufferUsage::CopyDst) ==
            (vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst));

    REQUIRE(ToVk(TextureUsage::None) == vk::ImageUsageFlags{});
    REQUIRE(ToVk(TextureUsage::Sampled) == vk::ImageUsageFlagBits::eSampled);
    REQUIRE(ToVk(TextureUsage::Storage) == vk::ImageUsageFlagBits::eStorage);
    REQUIRE(ToVk(TextureUsage::ColorAttachment) == vk::ImageUsageFlagBits::eColorAttachment);
    REQUIRE(ToVk(TextureUsage::DepthStencilAttachment) ==
            vk::ImageUsageFlagBits::eDepthStencilAttachment);
    REQUIRE(ToVk(TextureUsage::CopySrc) == vk::ImageUsageFlagBits::eTransferSrc);
    REQUIRE(ToVk(TextureUsage::CopyDst) == vk::ImageUsageFlagBits::eTransferDst);

    // The combination every loaded texture uses today.
    REQUIRE(ToVk(TextureUsage::CopyDst | TextureUsage::Sampled) ==
            (vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled));
}

TEST_CASE("A flags value carrying an unmapped bit is rejected rather than silently dropped",
          "[RhiConversions]")
{
    // Simulates adding an enumerator to the enum and its switch but forgetting
    // the kAll* array. Dropping the bit would produce a usage mask or barrier
    // quietly missing something, which is far harder to find than a throw.
    const auto unmappedBit = static_cast<BufferUsage>(1u << 20);

    REQUIRE_THROWS_AS(ToVk(unmappedBit), std::runtime_error);
    REQUIRE_THROWS_AS(ToVk(BufferUsage::Vertex | unmappedBit), std::runtime_error);
}

TEST_CASE("Flag operators behave as flags", "[RhiConversions]")
{
    REQUIRE_FALSE(Any(BufferUsage::None));
    REQUIRE(Any(BufferUsage::Vertex));

    constexpr BufferUsage both = BufferUsage::Vertex | BufferUsage::Index;
    REQUIRE(HasAll(both, BufferUsage::Vertex));
    REQUIRE(HasAll(both, BufferUsage::Index));
    REQUIRE(HasAll(both, both));
    REQUIRE_FALSE(HasAll(both, BufferUsage::Uniform));
    REQUIRE_FALSE(HasAll(BufferUsage::Vertex, both));

    BufferUsage accumulated = BufferUsage::None;
    accumulated |= BufferUsage::Storage;
    REQUIRE(HasAll(accumulated, BufferUsage::Storage));

    accumulated &= ~BufferUsage::Storage;
    REQUIRE_FALSE(Any(accumulated));

    // Distinct bits, so no two usages alias each other.
    REQUIRE_FALSE(Any(BufferUsage::Vertex & BufferUsage::Index));
}

TEST_CASE("Descs default to something inert rather than something plausible", "[RhiConversions]")
{
    // A default-constructed desc should be obviously unusable, not accidentally
    // valid: a desc that defaults to a real format and no usage is the kind of
    // thing that creates a resource nobody asked for.
    const BufferDesc buffer{};
    REQUIRE(buffer.Size == 0u);
    REQUIRE(buffer.Usage == BufferUsage::None);
    REQUIRE(buffer.Access == MemoryAccess::GpuOnly);

    const TextureDesc texture{};
    REQUIRE(texture.Format == Hikari::Rhi::Format::Undefined);
    REQUIRE(texture.Usage == TextureUsage::None);
    REQUIRE(texture.Dimension == TextureDimension::Texture2D);
    REQUIRE(texture.MipLevels == 1u);
    REQUIRE(texture.ArrayLayers == 1u);
    REQUIRE(texture.Samples == SampleCount::X1);
    REQUIRE_FALSE(texture.bCubeCompatible);

    // Extent3D's depth defaults to 1, not 0 — a 2D texture leaves it alone, and
    // a zero depth would be an invalid extent for both APIs.
    REQUIRE(texture.Extent == Extent3D{0u, 0u, 1u});

    // The sampler defaults are what the cloud pass's sampler is created from,
    // so it names no field at all; changing one of these silently changes how
    // that texture is filtered.
    const SamplerDesc sampler{};
    REQUIRE(sampler.MagFilter == Filter::Linear);
    REQUIRE(sampler.MinFilter == Filter::Linear);
    REQUIRE(sampler.MipmapFilter == MipmapMode::Linear);
    REQUIRE(sampler.AddressU == AddressMode::Repeat);
    REQUIRE(sampler.AddressV == AddressMode::Repeat);
    REQUIRE(sampler.AddressW == AddressMode::Repeat);
    REQUIRE(sampler.Border == BorderColor::OpaqueBlackInt);
    REQUIRE(sampler.MipLodBias == 0.f);
    REQUIRE(sampler.MinLod == 0.f);
    REQUIRE(sampler.MaxLod == 0.f);
    REQUIRE_FALSE(sampler.bAnisotropyEnable);
    REQUIRE_FALSE(sampler.bCompareEnable);

    // Both of the view desc's sentinels mean "follow the texture". They are
    // what stops every call site having to restate a format it does not choose
    // and an aspect it would get wrong for a depth texture.
    const TextureViewDesc view{};
    REQUIRE_FALSE(view.Texture.IsValid());
    REQUIRE(view.Dimension == TextureViewDimension::Texture2D);
    REQUIRE(view.Format == Hikari::Rhi::Format::Undefined);
    REQUIRE(view.Aspect == TextureAspect::None);
    REQUIRE(view.MipCount == 1u);
    REQUIRE(view.LayerCount == 1u);
}

TEST_CASE("A cube view is a view dimension, not a texture dimension", "[RhiConversions]")
{
    // The split is the whole reason TextureViewDimension exists: a cubemap is
    // stored as a 2D texture with six layers in both APIs, and only the view
    // calls it a cube. A single enum covering both would make "3D texture" and
    // "cube view" the same kind of thing, which they are not.
    REQUIRE(ToVk(TextureViewDimension::TextureCube) == vk::ImageViewType::eCube);
    REQUIRE(ToVk(TextureViewDimension::Texture2DArray) == vk::ImageViewType::e2DArray);
    REQUIRE(ToVk(TextureDimension::Texture2D) == vk::ImageType::e2D);
    REQUIRE(ToVk(TextureDimension::Texture3D) == vk::ImageType::e3D);
}

TEST_CASE("A barrier preset carries no texture until On() gives it one", "[RhiConversions]")
{
    // The presets are constants, which they can only be because they name no
    // resource. On() is the pairing, and it has to leave the description alone.
    constexpr TextureBarrier preset = BarrierPresets::CopyDstToShaderResource(6u);
    REQUIRE_FALSE(preset.Texture.IsValid());

    const TextureHandle texture = TextureHandle::FromIndexAndGeneration(7u, 3u);
    const TextureBarrier applied = preset.On(texture);

    REQUIRE(applied.Texture == texture);
    REQUIRE(applied.LayerCount == 6u);
    REQUIRE(applied.SrcStage == preset.SrcStage);
    REQUIRE(applied.SrcAccess == preset.SrcAccess);
    REQUIRE(applied.DstStage == preset.DstStage);
    REQUIRE(applied.DstAccess == preset.DstAccess);
    REQUIRE(applied.OldLayout == preset.OldLayout);
    REQUIRE(applied.NewLayout == preset.NewLayout);

    // The preset itself is untouched, so one can be applied to several textures.
    REQUIRE_FALSE(preset.Texture.IsValid());
}

TEST_CASE("Diagnostic severities map to the Vulkan tier of the same name", "[RhiConversions]")
{
    REQUIRE(ToVk(DiagnosticSeverity::Info) == vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo);
    REQUIRE(ToVk(DiagnosticSeverity::Warning) ==
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning);
    REQUIRE(ToVk(DiagnosticSeverity::Error) == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

    // Every neutral severity survives the round trip. Asserted in this direction
    // only: the reverse does not hold for eVerbose, which is the next case.
    for (const DiagnosticSeverity severity :
         {DiagnosticSeverity::Info, DiagnosticSeverity::Warning, DiagnosticSeverity::Error})
        REQUIRE(FromVk(ToVk(severity)) == severity);
}

TEST_CASE("A verbose Vulkan message is reported as Info, not dropped", "[RhiConversions]")
{
    // The neutral scale has no verbose tier. The mapping must collapse it into
    // Info rather than treating it as unknown: a caller that asked for Info-level
    // diagnostics silently losing the driver's most detailed messages is a
    // debugging trap, and the run report would show zero of them.
    REQUIRE(FromVk(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose) ==
            DiagnosticSeverity::Info);
    REQUIRE(FromVk(vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo) == DiagnosticSeverity::Info);
}

TEST_CASE("Severity ordering is preserved, so it can be used as a threshold",
          "[RhiConversions]")
{
    // The debug callback filters with `severity < ToVk(minimum)`, which is only
    // correct if the Vulkan enumerator values ascend with seriousness. The
    // specification says they do; this pins it, because the failure mode is
    // silent — an inverted comparison would either drop every message or drop
    // none, and the baseline run produces no diagnostics either way.
    REQUIRE(ToVk(DiagnosticSeverity::Info) < ToVk(DiagnosticSeverity::Warning));
    REQUIRE(ToVk(DiagnosticSeverity::Warning) < ToVk(DiagnosticSeverity::Error));

    // Verbose sits below the lowest neutral severity, so a minimum of Info
    // filters it out.
    REQUIRE(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose <
            ToVk(DiagnosticSeverity::Info));
}
