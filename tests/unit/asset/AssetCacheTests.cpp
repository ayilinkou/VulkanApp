#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>

#include <asset/AssetCache.h>

using Hikari::Asset::AssetCache;

namespace
{
/** Stands in for a loaded resource: the cache never looks inside one. */
struct Asset
{
    explicit Asset(int id) : Id(id) {}
    int Id = 0;
};

/** Counts how often a cache actually had to load, which is what a hit avoids. */
struct CountingLoader
{
    int Loads = 0;

    std::shared_ptr<Asset> operator()()
    {
        ++Loads;
        return std::make_shared<Asset>(Loads);
    }
};
} // namespace

TEST_CASE("A live entry is handed back instead of loaded again", "[AssetCache]")
{
    AssetCache<Asset> cache;
    CountingLoader loader;

    const std::shared_ptr<Asset> first = cache.Get("shared", std::ref(loader));
    const std::shared_ptr<Asset> second = cache.Get("shared", std::ref(loader));

    REQUIRE(first == second);
    REQUIRE(loader.Loads == 1);
    REQUIRE(cache.LiveCount() == 1);
}

TEST_CASE("Two caches share nothing, having no table between them", "[AssetCache]")
{
    AssetCache<Asset> first;
    AssetCache<Asset> second;
    CountingLoader firstLoader;
    CountingLoader secondLoader;

    // The same key in both: with a process-wide cache behind them the second
    // would find the first one's entry, which is exactly what a registry per
    // scene must not do.
    const std::shared_ptr<Asset> fromFirst = first.Get("model.gltf", std::ref(firstLoader));
    const std::shared_ptr<Asset> fromSecond = second.Get("model.gltf", std::ref(secondLoader));

    REQUIRE(fromFirst != fromSecond);
    REQUIRE(firstLoader.Loads == 1);
    REQUIRE(secondLoader.Loads == 1);
    REQUIRE(first.LiveCount() == 1);
    REQUIRE(second.LiveCount() == 1);
}

TEST_CASE("An entry whose resource has gone is loaded afresh", "[AssetCache]")
{
    AssetCache<Asset> cache;
    CountingLoader loader;

    {
        const std::shared_ptr<Asset> transient = cache.Get("temporary", std::ref(loader));
        REQUIRE(cache.LiveCount() == 1);
    }

    // The entry survives the resource, holding a weak reference to nothing, so
    // asking again has to load rather than hand back an empty pointer.
    REQUIRE(cache.LiveCount() == 0);

    const std::shared_ptr<Asset> reloaded = cache.Get("temporary", std::ref(loader));
    REQUIRE(reloaded != nullptr);
    REQUIRE(loader.Loads == 2);
}

TEST_CASE("Purging drops the expired entries and counts them", "[AssetCache]")
{
    AssetCache<Asset> cache;
    CountingLoader loader;

    const std::shared_ptr<Asset> kept = cache.Get("kept", std::ref(loader));
    {
        const std::shared_ptr<Asset> dropped = cache.Get("dropped", std::ref(loader));
    }

    REQUIRE(cache.Purge() == 1u);
    REQUIRE(cache.LiveCount() == 1);

    // Nothing expired is left, so a second purge finds nothing to do.
    REQUIRE(cache.Purge() == 0u);
}

TEST_CASE("A loader that fails leaves no entry behind", "[AssetCache]")
{
    AssetCache<Asset> cache;

    const std::shared_ptr<Asset> failed = cache.Get("missing", [] { return nullptr; });

    REQUIRE(failed == nullptr);
    REQUIRE(cache.LiveCount() == 0);

    // A null result must not be remembered: the next request has to try again
    // rather than report the same failure from the table.
    bool bLoaded = false;
    const std::shared_ptr<Asset> retried = cache.Get("missing",
                                                     [&bLoaded]
                                                     {
                                                         bLoaded = true;
                                                         return std::make_shared<Asset>(1);
                                                     });
    REQUIRE(bLoaded);
    REQUIRE(retried != nullptr);
}
