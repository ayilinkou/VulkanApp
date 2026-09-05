#pragma once

#include <cstdint>
#include <string>

#include <rhi/Handles.h>
#include <rhi/IDevice.h>
#include <rhi/TextureDesc.h>
#include <rhi/TextureViewDesc.h>
#include <rhi/UniqueHandle.h>

enum TextureBinding : uint8_t
{
    Albedo,
    Normal,
    MetallicRoughness,

    COUNT
};

/**
 * An image plus the view onto all of it, owned together.
 *
 * The pairing is what almost every user of a texture wants — an attachment, a
 * sampled material texture, a compute output — and keeping the two handles in
 * one place is what makes their destruction order right without each owner
 * having to know that a view must not outlive its image.
 *
 * Asset-side rather than part of the RHI: it carries the path it was loaded
 * from, which is what ResourceCache keys on. It moves into the Asset module in
 * Stage 7 along with the rest of the loading path.
 */
class Texture
{
public:
    Texture() = default;

    /**
     * Creates the image described by `desc` and a view covering every mip and
     * layer of it. The view's format and aspect follow the texture's, which is
     * what makes a depth texture get a depth view without the caller saying so.
     */
    Texture(Hikari::Rhi::IDevice& device, const Hikari::Rhi::TextureDesc& desc,
            Hikari::Rhi::TextureViewDimension viewDimension, std::string path = {});

    /**
     * Move-only, and the move assignment is written out rather than defaulted:
     * it has to release what it is overwriting in the reverse of declaration
     * order. See the definition.
     */
    Texture(Texture&&) noexcept = default;
    Texture& operator=(Texture&& other) noexcept;

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Hikari::Rhi::TextureHandle GetHandle() const { return m_Image.Get(); }
    Hikari::Rhi::TextureViewHandle GetView() const { return m_View.Get(); }

    const std::string& GetPath() const { return m_Path; }

private:
    /**
     * The view is declared second so that it is destroyed first. The
     * specification lets objects be destroyed in any order except parent before
     * child, and a view's parent is the device rather than the image — but it
     * also says an object passed in at creation "may be accessed by the
     * implementation any time that the created object is accessed", and that a
     * destroyed object "must not be accessed again, either directly or via
     * access through another object". Destroying the image first is therefore
     * at best subtle and is caught by no validation layer, so it is not done.
     */
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::TextureHandle> m_Image;
    Hikari::Rhi::UniqueHandle<Hikari::Rhi::TextureViewHandle> m_View;

    std::string m_Path;
};
