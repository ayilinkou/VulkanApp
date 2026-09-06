#include "vulkan/VulkanConversions.h"

#include <format>
#include <stdexcept>
#include <type_traits>

namespace Hikari::Rhi::Vulkan
{
vk::ShaderStageFlags ToVk(ShaderStage stages)
{
    vk::ShaderStageFlags result{};
    if (Any(stages & ShaderStage::Vertex))
        result |= vk::ShaderStageFlagBits::eVertex;
    if (Any(stages & ShaderStage::Pixel))
        result |= vk::ShaderStageFlagBits::eFragment;
    if (Any(stages & ShaderStage::Compute))
        result |= vk::ShaderStageFlagBits::eCompute;

    if (!result)
        throw std::runtime_error("Rhi::Vulkan::ToVk(ShaderStage): visible to no shader stage.");

    return result;
}

vk::AttachmentLoadOp ToVkLoadOp(LoadOp op)
{
    switch (op)
    {
        case LoadOp::Preserve:
            return vk::AttachmentLoadOp::eLoad;
        case LoadOp::Clear:
            return vk::AttachmentLoadOp::eClear;
        case LoadOp::Discard:
            return vk::AttachmentLoadOp::eDontCare;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVkLoadOp: unmapped LoadOp.");
}

vk::AttachmentStoreOp ToVkStoreOp(StoreOp op)
{
    switch (op)
    {
        case StoreOp::Preserve:
            return vk::AttachmentStoreOp::eStore;
        case StoreOp::Discard:
            return vk::AttachmentStoreOp::eDontCare;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVkStoreOp: unmapped StoreOp.");
}

namespace
{
/**
 * Folds a neutral flags value into its Vulkan equivalent one bit at a time.
 *
 * The per-bit function it is handed carries the switch, so the compile-time
 * "every enumerator is mapped" guarantee applies to flags enums as well as
 * scalar ones. The leftover check then catches the other half of that mistake:
 * a new bit added to the enum and its switch but not to the kAll* array would
 * otherwise be silently dropped, producing a barrier or a usage mask that is
 * quietly missing something. Here it throws instead.
 */
template <typename VkFlags, typename FlagEnum, size_t N, typename ToBitFn>
VkFlags ConvertFlags(FlagEnum value, const std::array<FlagEnum, N>& allBits, ToBitFn toBit,
                     const char* what)
{
    VkFlags result{};
    FlagEnum remaining = value;

    for (const FlagEnum bit : allBits)
    {
        if (!HasAll(value, bit))
            continue;

        result |= toBit(bit);
        remaining &= ~bit;
    }

    if (Any(remaining))
    {
        throw std::runtime_error(std::format(
            "Rhi::Vulkan::ToVk({}): bit(s) {:#x} are not in kAll{} — add the new "
            "enumerator to that array.",
            what, static_cast<uint64_t>(std::underlying_type_t<FlagEnum>(remaining)), what));
    }

    return result;
}

/**
 * Reverses a one-to-one mapping by searching the enumerator list, so that ToVk
 * stays the only place the mapping is written down.
 */
template <typename NeutralEnum, size_t N, typename VkValue>
NeutralEnum ConvertBack(VkValue value, const std::array<NeutralEnum, N>& all, const char* what)
{
    for (const NeutralEnum candidate : all)
    {
        if (ToVk(candidate) == value)
            return candidate;
    }

    throw std::runtime_error(std::format("Rhi::Vulkan::FromVk: Vulkan value {:#x} has no Rhi::{} "
                                         "equivalent.",
                                         static_cast<uint64_t>(value), what));
}

vk::PipelineStageFlags2 ToVkBit(PipelineStage stage)
{
    switch (stage)
    {
        // The sync2 replacement for the legacy eTopOfPipe / eBottomOfPipe pair: as
        // a source it means there is nothing to wait for, as a destination that
        // there is nothing to release to.
        case PipelineStage::None:
            return vk::PipelineStageFlagBits2::eNone;

        // D3D12_BARRIER_SYNC_DRAW covers the whole draw pipeline, which is what
        // eAllGraphics is. It therefore collapses onto the same Vulkan value as
        // AllGraphics — one of the reasons PipelineStage has no FromVk.
        case PipelineStage::Draw:
            return vk::PipelineStageFlagBits2::eAllGraphics;

        case PipelineStage::VertexStage:
            return vk::PipelineStageFlagBits2::eVertexShader;
        case PipelineStage::PixelStage:
            return vk::PipelineStageFlagBits2::eFragmentShader;
        case PipelineStage::ComputeStage:
            return vk::PipelineStageFlagBits2::eComputeShader;

        // Depth and stencil testing straddles two Vulkan stages: the test can run
        // before the fragment shader, or after it when the shader is what decides
        // the fragment's depth or coverage. Which one applies is a property of the
        // pipeline, not of the barrier, so a barrier has to name both — naming one
        // is a classic source of intermittent, driver-dependent corruption.
        //
        // These two stages also cover loading and storing a depth attachment,
        // which is why a depth attachment's load/store does not appear under
        // RenderTarget the way a colour attachment's does.
        case PipelineStage::DepthStencil:
            return vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                   vk::PipelineStageFlagBits2::eLateFragmentTests;

        case PipelineStage::RenderTarget:
            return vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        case PipelineStage::Copy:
            return vk::PipelineStageFlagBits2::eCopy;
        case PipelineStage::Resolve:
            return vk::PipelineStageFlagBits2::eResolve;
        case PipelineStage::AllGraphics:
            return vk::PipelineStageFlagBits2::eAllGraphics;
        case PipelineStage::All:
            return vk::PipelineStageFlagBits2::eAllCommands;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(PipelineStage): unhandled enumerator.");
}

vk::AccessFlags2 ToVkBit(AccessFlags access)
{
    switch (access)
    {
        case AccessFlags::None:
            return vk::AccessFlagBits2::eNone;
        case AccessFlags::VertexBufferRead:
            return vk::AccessFlagBits2::eVertexAttributeRead;
        case AccessFlags::IndexBufferRead:
            return vk::AccessFlagBits2::eIndexRead;
        case AccessFlags::ConstantBufferRead:
            return vk::AccessFlagBits2::eUniformRead;
        case AccessFlags::ShaderRead:
            return vk::AccessFlagBits2::eShaderRead;

        // D3D12_BARRIER_ACCESS_UNORDERED_ACCESS covers reads and writes as one
        // concept, so this widens to both Vulkan bits. Widening a barrier is safe —
        // it invalidates or flushes more than strictly needed — whereas splitting
        // the neutral enum in two would give the RHI a distinction D3D12 does not
        // have. If profiling ever shows the extra invalidation matters, that is the
        // point to reconsider, not now.
        case AccessFlags::UnorderedAccess:
            return vk::AccessFlagBits2::eShaderStorageRead |
                   vk::AccessFlagBits2::eShaderStorageWrite;

        case AccessFlags::RenderTargetRead:
            return vk::AccessFlagBits2::eColorAttachmentRead;
        case AccessFlags::RenderTargetWrite:
            return vk::AccessFlagBits2::eColorAttachmentWrite;
        case AccessFlags::DepthStencilRead:
            return vk::AccessFlagBits2::eDepthStencilAttachmentRead;
        case AccessFlags::DepthStencilWrite:
            return vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        case AccessFlags::CopySrc:
            return vk::AccessFlagBits2::eTransferRead;
        case AccessFlags::CopyDst:
            return vk::AccessFlagBits2::eTransferWrite;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(AccessFlags): unhandled enumerator.");
}

vk::BufferUsageFlags ToVkBit(BufferUsage usage)
{
    switch (usage)
    {
        case BufferUsage::None:
            return {};
        case BufferUsage::Vertex:
            return vk::BufferUsageFlagBits::eVertexBuffer;
        case BufferUsage::Index:
            return vk::BufferUsageFlagBits::eIndexBuffer;
        case BufferUsage::Uniform:
            return vk::BufferUsageFlagBits::eUniformBuffer;
        case BufferUsage::Storage:
            return vk::BufferUsageFlagBits::eStorageBuffer;
        case BufferUsage::CopySrc:
            return vk::BufferUsageFlagBits::eTransferSrc;
        case BufferUsage::CopyDst:
            return vk::BufferUsageFlagBits::eTransferDst;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(BufferUsage): unhandled enumerator.");
}

vk::ImageUsageFlags ToVkBit(TextureUsage usage)
{
    switch (usage)
    {
        case TextureUsage::None:
            return {};
        case TextureUsage::Sampled:
            return vk::ImageUsageFlagBits::eSampled;
        case TextureUsage::Storage:
            return vk::ImageUsageFlagBits::eStorage;
        case TextureUsage::ColorAttachment:
            return vk::ImageUsageFlagBits::eColorAttachment;
        case TextureUsage::DepthStencilAttachment:
            return vk::ImageUsageFlagBits::eDepthStencilAttachment;
        case TextureUsage::CopySrc:
            return vk::ImageUsageFlagBits::eTransferSrc;
        case TextureUsage::CopyDst:
            return vk::ImageUsageFlagBits::eTransferDst;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(TextureUsage): unhandled enumerator.");
}

vk::ImageAspectFlags ToVkBit(TextureAspect aspect)
{
    switch (aspect)
    {
        case TextureAspect::None:
            return {};
        case TextureAspect::Color:
            return vk::ImageAspectFlagBits::eColor;
        case TextureAspect::Depth:
            return vk::ImageAspectFlagBits::eDepth;
        case TextureAspect::Stencil:
            return vk::ImageAspectFlagBits::eStencil;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(TextureAspect): unhandled enumerator.");
}

/**
 * The capabilities that satisfy a role. Internal, because the test that uses it
 * is "any of these bits" and exposing the mask would invite "all of them" — see
 * FamilySupports in the header.
 */
vk::QueueFlags SatisfyingCapabilities(QueueType role)
{
    switch (role)
    {
        case QueueType::Graphics:
            return vk::QueueFlagBits::eGraphics;
        case QueueType::Compute:
            return vk::QueueFlagBits::eCompute;

        // Not just eTransfer: a family advertising graphics or compute may omit
        // the transfer bit while still being able to copy.
        case QueueType::Copy:
            return vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eGraphics |
                   vk::QueueFlagBits::eCompute;
    }

    throw std::runtime_error("Rhi::Vulkan::FamilySupports(): unhandled QueueType enumerator.");
}
} // namespace

vk::Format ToVk(Format format)
{
    switch (format)
    {
        case Format::Undefined:
            return vk::Format::eUndefined;
        case Format::R8Unorm:
            return vk::Format::eR8Unorm;
        case Format::RGBA8Unorm:
            return vk::Format::eR8G8B8A8Unorm;
        case Format::RGBA8Srgb:
            return vk::Format::eR8G8B8A8Srgb;
        case Format::BGRA8Unorm:
            return vk::Format::eB8G8R8A8Unorm;
        case Format::RGBA16Float:
            return vk::Format::eR16G16B16A16Sfloat;
        case Format::RG32Float:
            return vk::Format::eR32G32Sfloat;
        case Format::RGB32Float:
            return vk::Format::eR32G32B32Sfloat;
        case Format::RGBA32Float:
            return vk::Format::eR32G32B32A32Sfloat;
        case Format::D16Unorm:
            return vk::Format::eD16Unorm;
        case Format::D32Float:
            return vk::Format::eD32Sfloat;
        case Format::D24UnormS8Uint:
            return vk::Format::eD24UnormS8Uint;
        case Format::D32FloatS8Uint:
            return vk::Format::eD32SfloatS8Uint;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(Format): unhandled enumerator.");
}

Format FromVk(vk::Format format)
{
    return ConvertBack(format, kAllFormats, "Format");
}

vk::PresentModeKHR ToVk(PresentMode mode)
{
    switch (mode)
    {
        case PresentMode::Immediate:
            return vk::PresentModeKHR::eImmediate;
        case PresentMode::Mailbox:
            return vk::PresentModeKHR::eMailbox;
        case PresentMode::Fifo:
            return vk::PresentModeKHR::eFifo;
        case PresentMode::FifoRelaxed:
            return vk::PresentModeKHR::eFifoRelaxed;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(PresentMode): unhandled enumerator.");
}

PresentMode FromVk(vk::PresentModeKHR mode)
{
    return ConvertBack(mode, kAllPresentModes, "PresentMode");
}

vk::SampleCountFlagBits ToVk(SampleCount samples)
{
    switch (samples)
    {
        case SampleCount::X1:
            return vk::SampleCountFlagBits::e1;
        case SampleCount::X2:
            return vk::SampleCountFlagBits::e2;
        case SampleCount::X4:
            return vk::SampleCountFlagBits::e4;
        case SampleCount::X8:
            return vk::SampleCountFlagBits::e8;
        case SampleCount::X16:
            return vk::SampleCountFlagBits::e16;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(SampleCount): unhandled enumerator.");
}

SampleCount FromVk(vk::SampleCountFlagBits samples)
{
    return ConvertBack(samples, kAllSampleCounts, "SampleCount");
}

vk::ImageType ToVk(TextureDimension dimension)
{
    switch (dimension)
    {
        case TextureDimension::Texture2D:
            return vk::ImageType::e2D;
        case TextureDimension::Texture3D:
            return vk::ImageType::e3D;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(TextureDimension): unhandled enumerator.");
}

TextureDimension FromVk(vk::ImageType imageType)
{
    return ConvertBack(imageType, kAllTextureDimensions, "TextureDimension");
}

vk::ImageViewType ToVk(TextureViewDimension dimension)
{
    switch (dimension)
    {
        case TextureViewDimension::Texture2D:
            return vk::ImageViewType::e2D;
        case TextureViewDimension::Texture2DArray:
            return vk::ImageViewType::e2DArray;
        case TextureViewDimension::TextureCube:
            return vk::ImageViewType::eCube;
        case TextureViewDimension::Texture3D:
            return vk::ImageViewType::e3D;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(TextureViewDimension): unhandled enumerator.");
}

TextureViewDimension FromVk(vk::ImageViewType viewType)
{
    return ConvertBack(viewType, kAllTextureViewDimensions, "TextureViewDimension");
}

vk::Filter ToVk(Filter filter)
{
    switch (filter)
    {
        case Filter::Nearest:
            return vk::Filter::eNearest;
        case Filter::Linear:
            return vk::Filter::eLinear;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(Filter): unhandled enumerator.");
}

Filter FromVk(vk::Filter filter)
{
    return ConvertBack(filter, kAllFilters, "Filter");
}

vk::SamplerMipmapMode ToVk(MipmapMode mode)
{
    switch (mode)
    {
        case MipmapMode::Nearest:
            return vk::SamplerMipmapMode::eNearest;
        case MipmapMode::Linear:
            return vk::SamplerMipmapMode::eLinear;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(MipmapMode): unhandled enumerator.");
}

MipmapMode FromVk(vk::SamplerMipmapMode mode)
{
    return ConvertBack(mode, kAllMipmapModes, "MipmapMode");
}

vk::SamplerAddressMode ToVk(AddressMode mode)
{
    switch (mode)
    {
        case AddressMode::Repeat:
            return vk::SamplerAddressMode::eRepeat;
        case AddressMode::MirroredRepeat:
            return vk::SamplerAddressMode::eMirroredRepeat;
        case AddressMode::ClampToEdge:
            return vk::SamplerAddressMode::eClampToEdge;
        case AddressMode::ClampToBorder:
            return vk::SamplerAddressMode::eClampToBorder;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(AddressMode): unhandled enumerator.");
}

AddressMode FromVk(vk::SamplerAddressMode mode)
{
    return ConvertBack(mode, kAllAddressModes, "AddressMode");
}

vk::CompareOp ToVk(CompareOp op)
{
    switch (op)
    {
        case CompareOp::Never:
            return vk::CompareOp::eNever;
        case CompareOp::Less:
            return vk::CompareOp::eLess;
        case CompareOp::Equal:
            return vk::CompareOp::eEqual;
        case CompareOp::LessOrEqual:
            return vk::CompareOp::eLessOrEqual;
        case CompareOp::Greater:
            return vk::CompareOp::eGreater;
        case CompareOp::NotEqual:
            return vk::CompareOp::eNotEqual;
        case CompareOp::GreaterOrEqual:
            return vk::CompareOp::eGreaterOrEqual;
        case CompareOp::Always:
            return vk::CompareOp::eAlways;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(CompareOp): unhandled enumerator.");
}

CompareOp FromVk(vk::CompareOp op)
{
    return ConvertBack(op, kAllCompareOps, "CompareOp");
}

vk::BorderColor ToVk(BorderColor color)
{
    switch (color)
    {
        case BorderColor::TransparentBlackFloat:
            return vk::BorderColor::eFloatTransparentBlack;
        case BorderColor::OpaqueBlackFloat:
            return vk::BorderColor::eFloatOpaqueBlack;
        case BorderColor::OpaqueWhiteFloat:
            return vk::BorderColor::eFloatOpaqueWhite;
        case BorderColor::TransparentBlackInt:
            return vk::BorderColor::eIntTransparentBlack;
        case BorderColor::OpaqueBlackInt:
            return vk::BorderColor::eIntOpaqueBlack;
        case BorderColor::OpaqueWhiteInt:
            return vk::BorderColor::eIntOpaqueWhite;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(BorderColor): unhandled enumerator.");
}

BorderColor FromVk(vk::BorderColor color)
{
    return ConvertBack(color, kAllBorderColors, "BorderColor");
}

VmaMemoryParams ToVk(MemoryAccess access)
{
    switch (access)
    {
        // VMA_MEMORY_USAGE_AUTO with no host-access flag means VMA is free to place
        // this in device-local memory, which is the point.
        case MemoryAccess::GpuOnly:
            return VmaMemoryParams{};

        // SEQUENTIAL_WRITE tells VMA the CPU only ever writes forward, which lets
        // it choose write-combined memory. Reading back through such a mapping is
        // legal but pathologically slow, so GpuToCpu exists separately rather than
        // this being reused for readback.
        case MemoryAccess::CpuToGpu:
            return VmaMemoryParams{.Usage = VMA_MEMORY_USAGE_AUTO,
                                   .Flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                            VMA_ALLOCATION_CREATE_MAPPED_BIT};

        case MemoryAccess::GpuToCpu:
            return VmaMemoryParams{.Usage = VMA_MEMORY_USAGE_AUTO,
                                   .Flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                            VMA_ALLOCATION_CREATE_MAPPED_BIT};
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(MemoryAccess): unhandled enumerator.");
}

MemoryAccess FromVk(const VmaMemoryParams& params)
{
    for (const MemoryAccess candidate : kAllMemoryAccesses)
    {
        if (ToVk(candidate) == params)
            return candidate;
    }

    throw std::runtime_error(
        std::format("Rhi::Vulkan::FromVk: VMA usage {} with flags {:#x} has no Rhi::MemoryAccess "
                    "equivalent.",
                    static_cast<uint32_t>(params.Usage), static_cast<uint32_t>(params.Flags)));
}

vk::DebugUtilsMessageSeverityFlagBitsEXT ToVk(DiagnosticSeverity severity)
{
    switch (severity)
    {
        case DiagnosticSeverity::Info:
            return vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo;
        case DiagnosticSeverity::Warning:
            return vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning;
        case DiagnosticSeverity::Error:
            return vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    }

    throw std::runtime_error(std::format("Rhi::Vulkan::ToVk: unhandled Rhi::DiagnosticSeverity {}.",
                                         static_cast<uint32_t>(severity)));
}

DiagnosticSeverity FromVk(vk::DebugUtilsMessageSeverityFlagBitsEXT severity)
{
    switch (severity)
    {
        // Verbose has no neutral counterpart and collapses into Info. Dropping
        // it instead would silently discard messages a caller asked to see.
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
            return DiagnosticSeverity::Info;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
            return DiagnosticSeverity::Warning;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
            return DiagnosticSeverity::Error;
    }

    // Reached only if a future Vulkan version adds a severity bit. Treated as an
    // error rather than thrown on: this runs inside the driver's callback, where
    // an exception would propagate through C code, and losing a message is
    // better than that.
    return DiagnosticSeverity::Error;
}

bool FamilySupports(vk::QueueFlags familyCapabilities, QueueType role)
{
    // "Any of", not "all of" — see the header. A dedicated transfer family has
    // only eTransfer, and a graphics family may have only eGraphics, yet both
    // can serve Copy.
    return (familyCapabilities & SatisfyingCapabilities(role)) != vk::QueueFlags{};
}

vk::PipelineStageFlags2 ToVk(PipelineStage stages)
{
    return ConvertFlags<vk::PipelineStageFlags2>(
        stages, kAllPipelineStages, [](PipelineStage bit) { return ToVkBit(bit); },
        "PipelineStages");
}

vk::AccessFlags2 ToVk(AccessFlags access)
{
    return ConvertFlags<vk::AccessFlags2>(
        access, kAllAccessFlags, [](AccessFlags bit) { return ToVkBit(bit); }, "AccessFlags");
}

vk::ImageLayout ToVk(TextureLayout layout)
{
    switch (layout)
    {
        case TextureLayout::Undefined:
            return vk::ImageLayout::eUndefined;

        // Both map to eGeneral: Vulkan has no separate "usable by any queue"
        // layout, which is what makes TextureLayout one-way (see the header).
        case TextureLayout::Common:
        case TextureLayout::UnorderedAccess:
            return vk::ImageLayout::eGeneral;

        case TextureLayout::RenderTarget:
            return vk::ImageLayout::eColorAttachmentOptimal;
        case TextureLayout::ShaderResource:
            return vk::ImageLayout::eShaderReadOnlyOptimal;

        // The combined depth+stencil layouts rather than the separate depth-only
        // ones from VK_KHR_separate_depth_stencil_layouts. Both are legal for an
        // image whose format has only a depth aspect, and choosing the combined
        // form keeps this conversion independent of the format — which it has to
        // be, since a layout arrives here without one. If a depth-only layout ever
        // turns out to matter for a specific transition, that wants a format-aware
        // overload rather than a second neutral enumerator.
        //
        // Load-bearing detail, because the renderer still names the depth-only
        // layouts where it begins rendering and where it writes a descriptor,
        // and a barrier's old layout must be the layout the image is actually
        // in: the two spellings are not merely both legal, they are the same
        // layout. The specification says of DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        // "It is equivalent to VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL and
        // VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL", and the same of
        // DEPTH_STENCIL_READ_ONLY_OPTIMAL against the read-only pair (Vulkan
        // specification, Image Layouts). So mixing the spellings across a
        // transition is correct rather than tolerated.
        case TextureLayout::DepthStencilWrite:
            return vk::ImageLayout::eDepthStencilAttachmentOptimal;
        case TextureLayout::DepthStencilRead:
            return vk::ImageLayout::eDepthStencilReadOnlyOptimal;

        case TextureLayout::CopySrc:
            return vk::ImageLayout::eTransferSrcOptimal;
        case TextureLayout::CopyDst:
            return vk::ImageLayout::eTransferDstOptimal;
        case TextureLayout::Present:
            return vk::ImageLayout::ePresentSrcKHR;
    }

    throw std::runtime_error("Rhi::Vulkan::ToVk(TextureLayout): unhandled enumerator.");
}

vk::BufferUsageFlags ToVk(BufferUsage usage)
{
    return ConvertFlags<vk::BufferUsageFlags>(
        usage, kAllBufferUsages, [](BufferUsage bit) { return ToVkBit(bit); }, "BufferUsages");
}

vk::ImageUsageFlags ToVk(TextureUsage usage)
{
    return ConvertFlags<vk::ImageUsageFlags>(
        usage, kAllTextureUsages, [](TextureUsage bit) { return ToVkBit(bit); }, "TextureUsages");
}

vk::ImageAspectFlags ToVk(TextureAspect aspect)
{
    return ConvertFlags<vk::ImageAspectFlags>(
        aspect, kAllTextureAspects, [](TextureAspect bit) { return ToVkBit(bit); },
        "TextureAspects");
}
} // namespace Hikari::Rhi::Vulkan
