#include "CubemapLoader.h"

#include "Cubemap.h"
#include "stb_image.h"

#include <core/Log.h>

#include <span>

using namespace Hikari;
using namespace Hikari::Core;

constexpr LogCategory LogCubemapLoader("Cubemap Loader");

CubemapLoader::CubemapLoader(Rhi::IDevice& rhiDevice, Rhi::IUploadContext& uploadContext)
    : m_RhiDevice(rhiDevice), m_UploadContext(uploadContext)
{
}

std::shared_ptr<Cubemap> CubemapLoader::Load(const CubemapCreateInfo& createInfo)
{
    static constexpr uint32_t faceCount = Cubemap::kFaceCount;
    struct FaceData
    {
        std::array<stbi_uc*, faceCount> Pixels;
        int Width, Height, Channels;
    } faceData;

    for (size_t i = 0; i < faceCount; i++)
    {
        const std::string* facePath;

        // face order: +X, -X, +Y, -Y, +Z, -Z
        switch (i)
        {
            case 0:
                facePath = &createInfo.RightPath;
                break;
            case 1:
                facePath = &createInfo.LeftPath;
                break;
            case 2:
                facePath = &createInfo.TopPath;
                break;
            case 3:
                facePath = &createInfo.BottomPath;
                break;
            case 4:
                facePath = &createInfo.BackPath;
                break;
            case 5:
                facePath = &createInfo.FrontPath;
                break;
            default:
                throw std::runtime_error("Cubemap has only 6 faces!");
        }

        LogMsg(LogSeverity::Info, LogCubemapLoader, "Loading texture: {}", facePath->c_str());

        faceData.Pixels[i] = stbi_load(facePath->c_str(), &faceData.Width, &faceData.Height,
                                       &faceData.Channels, STBI_rgb_alpha);

        if (!faceData.Pixels[i])
            throw std::runtime_error(std::format("Failed to load texture: {}", facePath->c_str()));
    }

    const uint32_t width = static_cast<uint32_t>(faceData.Width);
    const uint32_t height = static_cast<uint32_t>(faceData.Height);
    const uint64_t faceSize = static_cast<uint64_t>(width) * height * 4u;

    auto cubemap =
        std::make_shared<Cubemap>(m_RhiDevice, createInfo, Core::Extent2D{width, height});

    // One upload naming all six layers, not six uploads: a texture has to reach
    // the context whole, or a staging-budget flush landing between two faces
    // would discard the ones already written (see IUploadContext::UploadTexture).
    // Packing them into one staging buffer is the context's job now, which is
    // why the faces are handed over as they were decoded.
    std::array<Rhi::TextureUpload, faceCount> faces;
    for (uint32_t i = 0; i < faceCount; i++)
    {
        faces[i] = Rhi::TextureUpload{
            .Data = std::span(reinterpret_cast<const std::byte*>(faceData.Pixels[i]), faceSize),
            .BaseLayer = i,
            .Extent = {width, height, 1u}};
    }

    m_UploadContext.UploadTexture(cubemap->GetHandle(), faces);

    for (size_t i = 0; i < faceCount; i++)
    {
        stbi_image_free(faceData.Pixels[i]);
        faceData.Pixels[i] = nullptr;
    }

    return cubemap;
}
