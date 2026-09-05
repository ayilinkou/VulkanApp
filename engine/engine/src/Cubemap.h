#pragma once

#include <string>

#include <rhi/Handles.h>
#include <rhi/IDevice.h>
#include <rhi/RhiTypes.h>

#include "Texture.h"

struct CubemapCreateInfo
{
    std::string RightPath = "";
    std::string LeftPath = "";
    std::string TopPath = "";
    std::string BottomPath = "";
    std::string FrontPath = "";
    std::string BackPath = "";
    std::string Name = "Cubemap";
    Hikari::Rhi::Format Format = Hikari::Rhi::Format::Undefined;

    std::string Key() const
    {
        return RightPath + LeftPath + TopPath + BottomPath + FrontPath + BackPath;
    }
};

/**
 * Six square faces in one texture, viewed as a cube.
 *
 * A Texture underneath, because that is what a cubemap is in both APIs: a 2D
 * texture with six array layers, plus a view that says "cube". Only the view
 * differs from any other texture, so only the view is worth a separate type.
 */
class Cubemap
{
public:
    Cubemap() = default;
    Cubemap(Hikari::Rhi::IDevice& device, const CubemapCreateInfo& createInfo,
            Hikari::Core::Extent2D faceExtent);

    Hikari::Rhi::TextureHandle GetHandle() const { return m_Texture.GetHandle(); }
    Hikari::Rhi::TextureViewHandle GetView() const { return m_Texture.GetView(); }

    const std::string& GetName() const { return m_CreateInfo.Name; }
    const CubemapCreateInfo& GetCreateInfo() const { return m_CreateInfo; }

    static constexpr uint32_t kFaceCount = 6u;

private:
    Texture m_Texture;
    CubemapCreateInfo m_CreateInfo{};
};
