#include "TextureLoader.h"

#include <span>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <core/Log.h>

using namespace Hikari;
using namespace Hikari::Core;

constexpr LogCategory LogTextureLoader("Texture Loader");

TextureLoader::TextureLoader(Rhi::IDevice& rhiDevice, Rhi::IUploadContext& uploadContext)
    : m_RhiDevice(rhiDevice), m_UploadContext(uploadContext)
{
}

std::shared_ptr<Texture> TextureLoader::Load(const std::string& path, const Rhi::Format format)
{
    LogMsg(LogSeverity::Info, LogTextureLoader, "Loading texture: {}", path.c_str());

    int width, height, channels;
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels)
    {
        LogMsg(LogSeverity::Error, LogTextureLoader, "Failed to load texture: {}", path.c_str());
        return nullptr;
    }

    const uint64_t imageSize = static_cast<uint64_t>(width) * height * 4u;
    std::shared_ptr<Texture> texture =
        CreateTextureFromPixels(pixels, static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                                format, imageSize, path);
    stbi_image_free(pixels);
    return texture;
}

std::shared_ptr<Texture>
TextureLoader::CreateTextureFromPixels(stbi_uc* pixels, const uint32_t width, const uint32_t height,
                                       const Rhi::Format format, const uint64_t size,
                                       const std::string& path)
{
    auto texture = std::make_shared<Texture>(
        m_RhiDevice,
        Rhi::TextureDesc{.Format = format,
                         .Extent = {width, height, 1u},
                         .Usage = Rhi::TextureUsage::Sampled | Rhi::TextureUsage::CopyDst,
                         .DebugName = path},
        Rhi::TextureViewDimension::Texture2D, path);

    // The context copies the pixels into staging before returning, so the
    // caller's stbi buffer can be freed as soon as this call is done. The
    // texture itself only holds them once a flush covering it has returned,
    // which the AssetRegistry does before handing the resource back.
    m_UploadContext.UploadTexture(
        texture->GetHandle(),
        Rhi::TextureUpload{.Data = std::span(reinterpret_cast<const std::byte*>(pixels), size),
                           .Extent = {width, height, 1u}});

    return texture;
}

std::shared_ptr<Texture> TextureLoader::LoadFallbackTexture(const Rhi::Format format)
{
    LogMsg(LogSeverity::Error, LogTextureLoader, "Loading fallback texture...");
    stbi_uc fallbackPixels[] = {255, 0, 255, 255};
    return CreateTextureFromPixels(fallbackPixels, 1u, 1u, format,
                                   sizeof(fallbackPixels) / sizeof(stbi_uc), "FallbackTexture");
}
