#pragma once

#include <cstdint>
#include <exception>
#include <memory>
#include <string>

#include <rhi/IDevice.h>
#include <rhi/RhiTypes.h>
#include <rhi/UploadContext.h>

#include <platform/Paths.h>

#include <assets/AssetCache.h>

#include <core/Log.h>

#include "CubemapLoader.h"
#include "ModelLoader.h"
#include "TextureLoader.h"

inline constexpr Hikari::Core::LogCategory LogAssetRegistry{"Asset Registry"};

struct CubemapCreateInfo;

class MaterialFactory;
class Texture;
class Cubemap;
class ModelData;

/**
 * Loads assets and hands out shared references to them, keyed by resolved path.
 *
 * Constructed rather than reached for: everything it needs arrives in the
 * constructor, and nothing finds it through a global. Two registries can exist
 * side by side with caches that know nothing of each other, which is what makes
 * a loader testable and what stops one scene's teardown reaching into another's.
 */
class AssetRegistry
{
public:
    /**
     * The material factory is passed through to ModelLoader rather than used
     * here: a model's materials are built while it loads, and the factory that
     * builds them belongs to the engine, not to this registry.
     */
    AssetRegistry(Hikari::Rhi::IDevice& rhiDevice, Hikari::Rhi::IUploadContext& uploadContext,
                  const Hikari::Platform::Paths& paths, MaterialFactory& materialFactory);
    ~AssetRegistry();

    AssetRegistry(const AssetRegistry&) = delete;
    AssetRegistry& operator=(const AssetRegistry&) = delete;
    AssetRegistry(AssetRegistry&&) = delete;
    AssetRegistry& operator=(AssetRegistry&&) = delete;

    /** Drops the entries whose resources have expired, and logs how many went. */
    void PurgeCaches();

    std::shared_ptr<Texture> LoadTexture(const std::string& filepath,
                                         const Hikari::Rhi::Format format);
    std::shared_ptr<Cubemap> LoadCubemap(const CubemapCreateInfo& createInfo);
    std::shared_ptr<ModelData> LoadModel(const std::string& modelPath);

private:
    /**
     * Flushes the upload context when the outermost load finishes.
     *
     * The nesting matters both ways. Loading a model loads its textures through
     * this same class, so flushing on every call would put each texture back in
     * its own submission and undo the batching entirely — Sponza's 77 became a
     * handful precisely because one model is one scope. And flushing when the
     * outermost one ends is what makes "a resource this class returns is on the
     * GPU" true by construction rather than by remembering.
     *
     * The run report's counters.run.uploadSubmissions is what guards this from a
     * distance: break the nesting and that number climbs with the scene's
     * texture count instead of staying at a handful.
     */
    class LoadScope
    {
    public:
        explicit LoadScope(AssetRegistry& owner) : m_Owner(owner) { ++m_Owner.m_LoadDepth; }

        /**
         * Flushing here can fail — it waits on the GPU — and a destructor that
         * throws while an exception from a failed load is already unwinding
         * terminates the process. Reported and swallowed instead, because by
         * this point the load has failed anyway and the useful error is the one
         * already in flight.
         */
        ~LoadScope()
        {
            if (--m_Owner.m_LoadDepth != 0u)
                return;

            try
            {
                m_Owner.m_UploadContext.Flush();
            }
            catch (const std::exception& error)
            {
                Hikari::Core::LogMsg(Hikari::Core::LogSeverity::Error, LogAssetRegistry,
                                     "Flushing pending uploads failed: {}", error.what());
            }
        }

        LoadScope(const LoadScope&) = delete;
        LoadScope& operator=(const LoadScope&) = delete;

    private:
        AssetRegistry& m_Owner;
    };

    Hikari::Rhi::IUploadContext& m_UploadContext;
    uint32_t m_LoadDepth = 0u;

    /**
     * Asset paths arrive here content-relative (a Model keeps the path it was
     * serialized with) and are resolved against the content root here, at the
     * point of loading.
     */
    const Hikari::Platform::Paths& m_Paths;

    /**
     * Owned rather than static: a loader has no state a second registry should
     * share, and its dependencies are this registry's own.
     */
    TextureLoader m_TextureLoader;
    CubemapLoader m_CubemapLoader;
    ModelLoader m_ModelLoader;

    Hikari::Assets::AssetCache<Texture> m_TextureCache;
    Hikari::Assets::AssetCache<Cubemap> m_CubemapCache;
    Hikari::Assets::AssetCache<ModelData> m_ModelCache;
};
