#include "Texture.h"

#include <utility>

using namespace Hikari;

Texture::Texture(Rhi::IDevice& device, const Rhi::TextureDesc& desc,
                 Rhi::TextureViewDimension viewDimension, std::string path)
    : m_Image(device, device.CreateTexture(desc)), m_Path(std::move(path))
{
    m_View = Rhi::UniqueHandle<Rhi::TextureViewHandle>(
        device,
        device.CreateTextureView(Rhi::TextureViewDesc{.Texture = m_Image.Get(),
                                                      .Dimension = viewDimension,
                                                      .MipCount = desc.MipLevels,
                                                      .LayerCount = desc.ArrayLayers,
                                                      .DebugName = desc.DebugName + " View"}));
}

Texture& Texture::operator=(Texture&& other) noexcept
{
    if (this != &other)
    {
        // The whole reason this is not defaulted: the implicit version assigns
        // members in declaration order, which would release the image before the
        // view that was made from it. Releasing both up front, view first, is
        // what keeps the ordering the destructor already has.
        m_View.Reset();
        m_Image.Reset();

        m_Image = std::move(other.m_Image);
        m_View = std::move(other.m_View);
        m_Path = std::move(other.m_Path);
    }
    return *this;
}
