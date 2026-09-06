#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

/**
 * The RHI's neutral vocabulary: the scalar types that appear in resource
 * descriptions and at the public API boundary. No Vulkan, no VMA, no D3D12.
 *
 * Every enum here is paired with a conversion table in the backend
 * (src/vulkan/VulkanConversions.h). Adding an enumerator means adding its
 * mapping in the same commit: the ToVk switches carry no `default:` label, so
 * the compiler rejects the build until the mapping exists.
 *
 * --- Why each enum also has a kAll* array ---
 *
 * C++ cannot iterate an enum's enumerators, so anything that needs to visit
 * every value has to be handed a list. Three things need that, and all three
 * would otherwise be hand-maintained copies of the enum:
 *
 *   1. Converting a flags value. ToVk(BufferUsage) has to decompose a bitmask
 *      into individual bits before it can map them, and looping over kAll*
 *      is the only way to ask "which bits are set" without hard-coding the
 *      list at the loop.
 *   2. Deriving the reverse mapping. FromVk searches for the neutral value
 *      whose ToVk matches, rather than being a second hand-written table that
 *      could disagree with the first.
 *   3. The tests. They iterate kAll* so that a new enumerator is covered the
 *      moment it is added, instead of being covered only if someone remembers
 *      to extend the test as well.
 *
 * Not every array serves all three: the scalar enums are not decomposed (1),
 * and the ones with no reverse mapping skip (2). QueueType is used only by the
 * tests. They exist uniformly anyway, because the cost is a line per
 * enumerator and the alternative is remembering which enums have one.
 *
 * Forgetting to add a new enumerator to its array is the one mistake here the
 * compiler cannot catch. For the flags enums it is caught at runtime instead —
 * ToVk throws on a bit it could not account for, rather than silently dropping
 * it (see ConvertFlags). For the rest it is a review item.
 */
namespace Hikari::Rhi
{
/**
 * Generates the bitwise operators a flags enum needs. Scoped enums have none
 * by default, which is what stops a flags enum being silently mixed with an
 * unrelated one.
 *
 * `Any` and `HasAll` exist because `flags & mask` yields the enum type, not a
 * bool, and `if (flags & mask)` therefore does not compile — deliberately, so
 * the intent (any bit vs every bit) has to be written down.
 */
#define RHI_DEFINE_FLAG_OPERATORS(EnumType)                                                        \
    constexpr EnumType operator|(EnumType a, EnumType b)                                           \
    {                                                                                              \
        using U = std::underlying_type_t<EnumType>;                                                \
        return static_cast<EnumType>(static_cast<U>(a) | static_cast<U>(b));                       \
    }                                                                                              \
    constexpr EnumType operator&(EnumType a, EnumType b)                                           \
    {                                                                                              \
        using U = std::underlying_type_t<EnumType>;                                                \
        return static_cast<EnumType>(static_cast<U>(a) & static_cast<U>(b));                       \
    }                                                                                              \
    constexpr EnumType operator^(EnumType a, EnumType b)                                           \
    {                                                                                              \
        using U = std::underlying_type_t<EnumType>;                                                \
        return static_cast<EnumType>(static_cast<U>(a) ^ static_cast<U>(b));                       \
    }                                                                                              \
    constexpr EnumType operator~(EnumType a)                                                       \
    {                                                                                              \
        using U = std::underlying_type_t<EnumType>;                                                \
        return static_cast<EnumType>(~static_cast<U>(a));                                          \
    }                                                                                              \
    constexpr EnumType& operator|=(EnumType& a, EnumType b)                                        \
    {                                                                                              \
        return a = a | b;                                                                          \
    }                                                                                              \
    constexpr EnumType& operator&=(EnumType& a, EnumType b)                                        \
    {                                                                                              \
        return a = a & b;                                                                          \
    }                                                                                              \
    constexpr EnumType& operator^=(EnumType& a, EnumType b)                                        \
    {                                                                                              \
        return a = a ^ b;                                                                          \
    }                                                                                              \
    constexpr bool Any(EnumType a)                                                                 \
    {                                                                                              \
        return static_cast<std::underlying_type_t<EnumType>>(a) != 0;                              \
    }                                                                                              \
    constexpr bool HasAll(EnumType value, EnumType mask)                                           \
    {                                                                                              \
        return (value & mask) == mask;                                                             \
    }

/**
 * Pixel formats. Curated rather than a mirror of VkFormat: every entry has both
 * a VkFormat and a DXGI_FORMAT equivalent, so this list is a
 * portability promise and not just a convenience.
 *
 * Two deliberate omissions, both from the depth-format candidate list the app
 * currently searches in FindDepthFormat:
 *
 *   * D16UnormS8Uint has no DXGI equivalent. dxgiformat.h offers stencil only
 *     alongside 24-bit unorm or 32-bit float depth; the enum runs straight
 *     from DXGI_FORMAT_D16_UNORM to DXGI_FORMAT_R16_UNORM. A depth+stencil
 *     format at 16-bit depth cannot be expressed. It was the last candidate the
 *     renderer's depth-format search tried, and dropping it costs nothing —
 *     see FindDepthFormat for why no conformant device could reach it.
 *   * The vertex-attribute formats (R32G32B32A32Sfloat and friends) are absent
 *     because vertex input stays Vulkan-side for the whole of Stage 5 (D8).
 *     They are portable and belong here when pipeline creation is neutralized.
 */
enum class Format : uint32_t
{
    Undefined = 0,

    R8Unorm,
    RGBA8Unorm,
    RGBA8Srgb,
    BGRA8Unorm,
    RGBA16Float,

    /**
     * Vertex attribute formats. In the same enum as the texture formats because
     * both APIs put them there -- one VkFormat, one DXGI_FORMAT -- and splitting
     * them would invent a distinction neither backend can act on.
     */
    RG32Float,
    RGB32Float,
    RGBA32Float,

    D16Unorm,
    D32Float,
    D24UnormS8Uint,
    D32FloatS8Uint,
};

inline constexpr std::array kAllFormats{
    Format::Undefined,      Format::R8Unorm,     Format::RGBA8Unorm, Format::RGBA8Srgb,
    Format::BGRA8Unorm,     Format::RGBA16Float, Format::RG32Float,  Format::RGB32Float,
    Format::RGBA32Float,    Format::D16Unorm,    Format::D32Float,   Format::D24UnormS8Uint,
    Format::D32FloatS8Uint,
};

/**
 * Whether `format` carries a depth component, and so needs a depth aspect
 * rather than a colour one when it appears in a barrier or a view.
 */
constexpr bool IsDepthFormat(Format format)
{
    return format == Format::D16Unorm || format == Format::D32Float ||
           format == Format::D24UnormS8Uint || format == Format::D32FloatS8Uint;
}

/**
 * Whether `format` also carries a stencil component. Separate from
 * IsDepthFormat because the stencil aspect has to be named explicitly in a
 * subresource range, and getting it wrong is a validation error rather than a
 * visible one.
 */
constexpr bool HasStencilComponent(Format format)
{
    return format == Format::D24UnormS8Uint || format == Format::D32FloatS8Uint;
}

/**
 * The size of one texel of `format` in a tightly packed buffer, or 0 where
 * there is no single answer.
 *
 * Exists because sizing a buffer for a copy is impossible without it: a
 * readback allocates Width * Height * this, and both APIs leave the caller to
 * work it out. Named for a texel rather than a block because every format in
 * the list above is uncompressed and one texel wide; a block-compressed format
 * added here would need a different question asked of it.
 *
 * Zero for three of them, and it is not a failure code so much as the honest
 * answer to a question that has none:
 *
 *   * Undefined names no memory at all.
 *   * The two combined depth/stencil formats have a size per *aspect*, and the
 *     two differ — a copy names one aspect, so the caller that knows which one
 *     is the only one that can size the buffer. D24UnormS8Uint is 4 bytes of
 *     depth and 1 of stencil; D32FloatS8Uint is 4 and 1.
 *
 * A caller that sizes a buffer from this therefore has to reject 0 rather than
 * allocate nothing, which is what makes the sentinel visible instead of
 * silently producing an empty copy.
 */
constexpr uint32_t BytesPerTexel(Format format)
{
    // No default label, so adding a Format that is not sized here fails the
    // build rather than falling through to a plausible number — the same deal
    // the conversion tables make.
    switch (format)
    {
        case Format::Undefined:
        case Format::D24UnormS8Uint:
        case Format::D32FloatS8Uint:
            return 0u;

        case Format::R8Unorm:
            return 1u;

        case Format::D16Unorm:
            return 2u;

        case Format::RGBA8Unorm:
        case Format::RGBA8Srgb:
        case Format::BGRA8Unorm:
        case Format::D32Float:
            return 4u;

        case Format::RGBA16Float:
        case Format::RG32Float:
            return 8u;

        case Format::RGB32Float:
            return 12u;

        case Format::RGBA32Float:
            return 16u;
    }

    return 0u;
}

/**
 * Which parts of a texture a barrier or a view refers to. Kept separate from
 * Format because a depth/stencil format has two aspects and an operation
 * usually names one of them.
 */
enum class TextureAspect : uint32_t
{
    None = 0,
    Color = 1 << 0,
    Depth = 1 << 1,
    Stencil = 1 << 2,
};
RHI_DEFINE_FLAG_OPERATORS(TextureAspect)

inline constexpr std::array kAllTextureAspects{
    TextureAspect::Color,
    TextureAspect::Depth,
    TextureAspect::Stencil,
};

/**
 * The aspect mask a barrier or view should use for `format`, so that the
 * depth/stencil decision is made in one place rather than at each call site.
 */
constexpr TextureAspect DefaultAspect(Format format)
{
    if (!IsDepthFormat(format))
        return TextureAspect::Color;

    return HasStencilComponent(format) ? (TextureAspect::Depth | TextureAspect::Stencil)
                                       : TextureAspect::Depth;
}

/**
 * The *role* work is submitted for, not a description of a queue. Chosen to
 * match D3D12's DIRECT / COMPUTE / COPY command list types, which have no
 * notion of a queue family index (plan D6); Vulkan's family indices stay
 * inside the backend.
 *
 * Nothing here says the three roles map to three distinct queues. A Vulkan
 * queue family advertises a mask of capabilities, and a "universal" family
 * with graphics + compute + transfer is guaranteed to exist on any device that
 * supports graphics at all — so one queue may back all three roles, which is
 * what happens on a device with nothing better. The backend is free to alias
 * them, and equally free to stop: uploads are submitted for the Copy role, and
 * land on a queue of their own wherever the device has one.
 *
 * Presentation is deliberately not a role here. It is not a queue capability
 * in Vulkan — support is a property of a (family, surface) pair, queried
 * separately — and D3D12 presents from a direct queue with no notion of a
 * present queue at all. So "can this queue present" is a question only the
 * present path can answer, and it stays behind that seam (plan D5): Stage 6's
 * IPresentTarget owns it.
 */
enum class QueueType : uint8_t
{
    Graphics = 0,
    Compute,
    Copy,
};

inline constexpr std::array kAllQueueTypes{
    QueueType::Graphics,
    QueueType::Compute,
    QueueType::Copy,
};

/**
 * Where a resource's memory lives and which side writes it. Maps onto VMA's
 * usage plus its host-access allocation flags, and onto D3D12's DEFAULT /
 * UPLOAD / READBACK heap types.
 */
enum class MemoryAccess : uint8_t
{
    /** Device-local, never mapped. The destination of a staged upload. */
    GpuOnly = 0,

    /**
     * Host-visible and written sequentially by the CPU, then read by the GPU:
     * staging buffers, and the persistently mapped uniform and instance
     * buffers. Sequential write is the important half — VMA may place this in
     * write-combined memory, where a read-modify-write from the CPU is
     * pathologically slow rather than merely uncached.
     */
    CpuToGpu,

    /**
     * Host-visible and read back by the CPU after the GPU has written it: the
     * screenshot staging buffer. Random access, so not write-combined.
     */
    GpuToCpu,
};

inline constexpr std::array kAllMemoryAccesses{
    MemoryAccess::GpuOnly,
    MemoryAccess::CpuToGpu,
    MemoryAccess::GpuToCpu,
};

/**
 * How a presented frame reaches the display.
 *
 * Vulkan is the API that names these, so its terms stand (D13): D3D12 spells
 * the same behaviour as a SyncInterval plus ALLOW_TEARING, and has no single
 * name to borrow. Only Fifo is guaranteed by the Vulkan specification — a
 * surface need not offer any of the others, which is why the default is a
 * preference rather than a requirement.
 */
enum class PresentMode : uint8_t
{
    Immediate,
    Mailbox,
    Fifo,
    FifoRelaxed,
};

inline constexpr std::array kAllPresentModes{
    PresentMode::Immediate,
    PresentMode::Mailbox,
    PresentMode::Fifo,
    PresentMode::FifoRelaxed,
};

/**
 * Values are the sample counts themselves, so a count can be converted to a
 * bit position arithmetically rather than through another table.
 */
enum class SampleCount : uint8_t
{
    X1 = 1,
    X2 = 2,
    X4 = 4,
    X8 = 8,
    X16 = 16,
};

inline constexpr std::array kAllSampleCounts{
    SampleCount::X1, SampleCount::X2, SampleCount::X4, SampleCount::X8, SampleCount::X16,
};

} // namespace Hikari::Rhi
