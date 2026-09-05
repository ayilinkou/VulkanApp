#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rhi/IDevice.h>
#include <rhi/UploadContext.h>

#include "Material.h"

struct aiScene;

class AssetRegistry;
class ModelData;

/** Owned by the AssetRegistry that loads through it; it caches nothing itself. */
class ModelLoader
{
public:
    /**
     * Holds the registry that owns it, because a model's materials load their
     * own textures — through that same registry, so that they land in one cache
     * and inside the caller's load scope rather than in a flush of their own.
     */
    ModelLoader(Hikari::Rhi::IDevice& rhiDevice, Hikari::Rhi::IUploadContext& uploadContext,
                AssetRegistry& assets);

    [[nodiscard]] std::shared_ptr<ModelData> Load(const std::string& path);

private:
    std::vector<std::unique_ptr<Material>> LoadMaterials(const aiScene* pScene,
                                                         const std::string& modelRoot);

    Hikari::Rhi::IDevice& m_RhiDevice;
    Hikari::Rhi::IUploadContext& m_UploadContext;
    AssetRegistry& m_Assets;
};
