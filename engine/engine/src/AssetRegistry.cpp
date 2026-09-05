#include "AssetRegistry.h"

#include <core/Log.h>
#include <core/MyMacros.h>

#include "Cubemap.h"
#include "ModelData.h"
#include "Texture.h"

using namespace Hikari;
using namespace Hikari::Core;
using namespace Hikari::Platform;

constexpr std::string_view fallbackTexturePrefix = "FallbackTexture";

AssetRegistry::AssetRegistry(Rhi::IDevice& rhiDevice, Rhi::IUploadContext& uploadContext,
                             const Paths& paths, MaterialFactory& materialFactory)
    : m_UploadContext(uploadContext), m_Paths(paths), m_TextureLoader(rhiDevice, uploadContext),
      m_CubemapLoader(rhiDevice, uploadContext),
      m_ModelLoader(rhiDevice, uploadContext, *this, materialFactory)
{
    LogMsg(LogSeverity::Info, LogAssetRegistry, "Constructed");
}

AssetRegistry::~AssetRegistry()
{
    LogMsg(LogSeverity::Info, LogAssetRegistry, "Destroyed");

    // Every entry is weak, so a live one means a resource outliving the registry
    // that loaded it — and its GPU handles outliving the device that owns them.
    assert(m_TextureCache.LiveCount() == 0);
    assert(m_CubemapCache.LiveCount() == 0);
    assert(m_ModelCache.LiveCount() == 0);
}

void AssetRegistry::PurgeCaches()
{
    uint32_t count = 0u;
    count += m_TextureCache.Purge();
    count += m_CubemapCache.Purge();
    count += m_ModelCache.Purge();

    if (count > 0u)
        LogMsg(LogSeverity::Info, LogAssetRegistry, "Purged {} expired resource entries.", count);
}

std::shared_ptr<Texture> AssetRegistry::LoadTexture(const std::string& filepath,
                                                    const Rhi::Format format)
{
    LoadScope uploads(*this);

    // Keyed on the resolved path so that the same file requested relatively
    // (from a scene) and absolutely (from a model's own texture references)
    // shares one cache entry.
    const std::string resolved = m_Paths.Content(filepath).string();
    const std::string key = resolved + std::to_string(static_cast<uint32_t>(format));
    auto tex = m_TextureCache.Get(key, [&] { return m_TextureLoader.Load(resolved, format); });
    if (!tex)
    {
        tex.reset();
        const std::string fallbackTextureKey =
            std::string(fallbackTexturePrefix) + std::to_string(static_cast<uint32_t>(format));
        return m_TextureCache.Get(fallbackTextureKey,
                                  [&] { return m_TextureLoader.LoadFallbackTexture(format); });
    }
    return tex;
}

std::shared_ptr<Cubemap> AssetRegistry::LoadCubemap(const CubemapCreateInfo& createInfo)
{
    LoadScope uploads(*this);
    return m_CubemapCache.Get(createInfo.Key(), [&] { return m_CubemapLoader.Load(createInfo); });
}

std::shared_ptr<ModelData> AssetRegistry::LoadModel(const std::string& modelPath)
{
    LoadScope uploads(*this);

    // ModelLoader derives its texture directory from the path it is given, so
    // handing it the resolved path is also what makes the model's own texture
    // references resolve.
    const std::string resolved = m_Paths.Content(modelPath).string();
    return m_ModelCache.Get(resolved, [&] { return m_ModelLoader.Load(resolved); });
}
