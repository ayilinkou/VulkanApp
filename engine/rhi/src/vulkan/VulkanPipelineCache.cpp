#include "vulkan/VulkanPipelineCache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <core/Log.h>

#include "vulkan/DebugNames.h"

namespace Hikari::Rhi::Vulkan
{
constexpr Core::LogCategory LogRhi("RHI");
namespace
{

/**
 * The pipeline cache header is 32 bytes with every field written least
 * significant byte first, whatever the host's byte order is, and the C standard
 * does not promise VkPipelineCacheHeaderVersionOne is packed to match. So it is
 * read a byte at a time. Vulkan spec, "Pipeline Cache Header".
 */
constexpr size_t kHeaderSize = 32;
constexpr size_t kVendorIdOffset = 8;
constexpr size_t kDeviceIdOffset = 12;
constexpr size_t kUuidOffset = 16;

uint32_t ReadLittleEndian32(std::span<const std::byte> bytes, size_t offset)
{
    return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

/**
 * Whether `data` is cache data this device can be seeded with.
 *
 * The header exists so that an application can answer this before handing the
 * blob over, and answering it is not optional:
 * VUID-VkPipelineCacheCreateInfo-initialDataSize-00769 makes passing anything
 * that did not come from vkGetPipelineCacheData invalid usage. The
 * implementation is required to ignore data it does not recognise, so the cost
 * of getting it wrong is not a crash — but a knowingly invalid call is one the
 * validation layers are entitled to report, and validationErrors is a number
 * this project keeps at zero.
 */
bool IsUsableCacheData(std::span<const std::byte> data,
                       const vk::PhysicalDeviceProperties& deviceProperties)
{
    if (data.size() < kHeaderSize)
    {
        Core::LogMsg(Core::LogSeverity::Warning, LogRhi,
                     "Pipeline cache file is {} bytes, too short to hold a header. Ignoring it.",
                     data.size());
        return false;
    }

    const uint32_t headerSize = ReadLittleEndian32(data, 0);
    const uint32_t headerVersion = ReadLittleEndian32(data, 4);

    if (headerSize != kHeaderSize ||
        headerVersion != static_cast<uint32_t>(vk::PipelineCacheHeaderVersion::eOne))
    {
        Core::LogMsg(Core::LogSeverity::Warning, LogRhi,
                     "Pipeline cache file has no valid header (size {}, version {}). Ignoring it.",
                     headerSize, headerVersion);
        return false;
    }

    // A different GPU, or the same one behind a driver that compiles
    // differently. Routine rather than suspect: the file is doing its job by
    // saying so.
    const uint32_t vendorId = ReadLittleEndian32(data, kVendorIdOffset);
    const uint32_t deviceId = ReadLittleEndian32(data, kDeviceIdOffset);
    const auto uuid = data.subspan(kUuidOffset, VK_UUID_SIZE);

    const bool bMatchesDevice =
        vendorId == deviceProperties.vendorID && deviceId == deviceProperties.deviceID &&
        std::ranges::equal(uuid, std::as_bytes(std::span(deviceProperties.pipelineCacheUUID)));

    if (!bMatchesDevice)
    {
        Core::LogMsg(
            Core::LogSeverity::Info, LogRhi,
            "Pipeline cache file was written by a different device or driver. Starting empty.");
        return false;
    }

    return true;
}

std::vector<std::byte> ReadCacheFile(const std::filesystem::path& path)
{
    // No file is the normal first-run case, so this reports "nothing to seed
    // with" rather than throwing the way Platform::ReadFile() does.
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return {};

    const std::streamoff size = file.tellg();
    if (size <= 0)
        return {};

    std::vector<std::byte> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), size);
    if (!file)
        return {};

    return data;
}
} // namespace

VulkanPipelineCache::VulkanPipelineCache(vk::raii::Device& device,
                                         const vk::PhysicalDeviceProperties& deviceProperties,
                                         const PipelineCacheDesc& desc)
    : m_Path(desc.Path)
{
    std::vector<std::byte> initialData;
    if (!m_Path.empty())
    {
        initialData = ReadCacheFile(m_Path);
        if (!initialData.empty() && !IsUsableCacheData(initialData, deviceProperties))
            initialData.clear();
    }

    // No VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT: without it the
    // implementation synchronises the cache itself, which is what lets several
    // threads create pipelines against one cache. Setting it would make that
    // the caller's problem in exchange for a lock this application does not
    // measure.
    const vk::PipelineCacheCreateInfo createInfo{.initialDataSize = initialData.size(),
                                                 .pInitialData = initialData.data()};
    m_Cache = vk::raii::PipelineCache(device, createInfo);

    if (!desc.DebugName.empty())
        SetVkDebugName(device, *m_Cache, vk::ObjectType::ePipelineCache, desc.DebugName.c_str());

    if (!initialData.empty())
    {
        Core::LogMsg(Core::LogSeverity::Info, LogRhi, "Seeded pipeline cache with {} bytes from {}",
                     initialData.size(), m_Path.string());
    }
    else if (m_Path.empty())
    {
        Core::LogMsg(Core::LogSeverity::Info, LogRhi,
                     "Pipeline cache is memory-only; it will not be saved");
    }
    else
    {
        Core::LogMsg(Core::LogSeverity::Info, LogRhi,
                     "Pipeline cache starting empty; will be saved to {}", m_Path.string());
    }
}

bool VulkanPipelineCache::Save()
{
    if (m_Path.empty())
        return false;

    const std::vector<uint8_t> data = m_Cache.getData();
    if (data.empty())
        return false;

    // Paths creates the user data directory at startup, so this only matters if
    // it went away while the application was running. No existence test in front
    // of it: create_directories already returns cleanly when the directory is
    // there, and testing first would only add a window for the answer to change.
    // The guard is for the other case — parent_path() is empty for a bare
    // filename, and create_directories("") is an error rather than a no-op.
    std::error_code ec;
    if (m_Path.has_parent_path())
        std::filesystem::create_directories(m_Path.parent_path(), ec);

    // Written beside the real file and moved over it, because a blob truncated
    // by a crash or a full disk would still carry a valid header and so would
    // pass every check the next run makes — and the size it claims is part of
    // what makes the data valid to hand back
    // (VUID-VkPipelineCacheCreateInfo-initialDataSize-00768). The rename is
    // atomic within a filesystem, so the next run sees either the old file or
    // the whole new one.
    std::filesystem::path tempPath = m_Path;
    tempPath += ".tmp";

    {
        std::ofstream file(tempPath, std::ios::binary | std::ios::trunc);
        if (file)
            file.write(reinterpret_cast<const char*>(data.data()),
                       static_cast<std::streamsize>(data.size()));

        if (!file)
        {
            Core::LogMsg(Core::LogSeverity::Warning, LogRhi, "Failed to write pipeline cache to {}",
                         tempPath.string());
            std::filesystem::remove(tempPath, ec);
            return false;
        }
    }

    std::filesystem::rename(tempPath, m_Path, ec);
    if (ec)
    {
        Core::LogMsg(Core::LogSeverity::Warning, LogRhi, "Failed to replace pipeline cache {}: {}",
                     m_Path.string(), ec.message());
        std::filesystem::remove(tempPath, ec);
        return false;
    }

    Core::LogMsg(Core::LogSeverity::Info, LogRhi, "Saved {} bytes of pipeline cache to {}",
                 data.size(), m_Path.string());
    return true;
}

VulkanPipelineCache& ToVulkan(IPipelineCache& cache)
{
    // dynamic_cast for the reason the native accessors use one: a wrong backend
    // here is undefined behaviour rather than a diagnosable error, and this
    // happens a handful of times at startup.
    auto* pVulkanCache = dynamic_cast<VulkanPipelineCache*>(&cache);
    if (!pVulkanCache)
        throw std::runtime_error("A non-Vulkan pipeline cache was given to Vulkan pipeline "
                                 "creation!");

    return *pVulkanCache;
}

vk::Optional<const vk::raii::PipelineCache> GetVkPipelineCache(IPipelineCache* pCache)
{
    if (!pCache)
        return nullptr;

    return ToVulkan(*pCache).Get();
}
} // namespace Hikari::Rhi::Vulkan
