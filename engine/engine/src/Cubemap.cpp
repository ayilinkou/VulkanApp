#include "Cubemap.h"

#include <format>

using namespace Hikari;

Cubemap::Cubemap(Rhi::IDevice& device, const CubemapCreateInfo& createInfo,
                 Core::Extent2D faceExtent)
    : m_Texture(device,
                Rhi::TextureDesc{.Format = createInfo.Format,
                                 .Extent = {faceExtent.Width, faceExtent.Height, 1u},
                                 .ArrayLayers = kFaceCount,
                                 .Usage = Rhi::TextureUsage::Sampled | Rhi::TextureUsage::CopyDst,
                                 .bCubeCompatible = true,
                                 .DebugName = std::format("{} Cubemap", createInfo.Name)},
                Rhi::TextureViewDimension::TextureCube),
      m_CreateInfo(createInfo)
{
}
