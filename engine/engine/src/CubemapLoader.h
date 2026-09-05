#pragma once

#include <memory>

#include <rhi/IDevice.h>
#include <rhi/UploadContext.h>

struct CubemapCreateInfo;

class Cubemap;

/** Owned by the AssetRegistry that loads through it; it caches nothing itself. */
class CubemapLoader
{
public:
    CubemapLoader(Hikari::Rhi::IDevice& rhiDevice, Hikari::Rhi::IUploadContext& uploadContext);

    [[nodiscard]] std::shared_ptr<Cubemap> Load(const CubemapCreateInfo& createInfo);

private:
    Hikari::Rhi::IDevice& m_RhiDevice;
    Hikari::Rhi::IUploadContext& m_UploadContext;
};
