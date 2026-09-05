#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Hikari::Asset
{

/**
 * A path-keyed cache of shared resources that owns none of them.
 *
 * Entries are weak, so a resource lives exactly as long as its users: the cache
 * hands out a second reference to one that is still alive, and forgets one that
 * is not. That is what makes two caches independent — there is no process-wide
 * table behind them, and a resource loaded through one is invisible to the
 * other unless its key is looked up there too.
 */
template <typename T>
class AssetCache
{
public:
    template <typename LoadFn>
    std::shared_ptr<T> Get(const std::string& key, LoadFn&& load)
    {
        std::lock_guard lock(m_Mutex);

        if (auto it = m_Cache.find(key); it != m_Cache.end())
        {
            if (std::shared_ptr<T> sp = it->second.lock())
                return sp;
            m_Cache.erase(it);
        }

        std::shared_ptr<T> sp = load();
        if (sp)
            m_Cache.emplace(key, std::weak_ptr(sp));
        return sp;
    }

    /** Removes the entries whose resources have expired, and says how many went. */
    uint32_t Purge()
    {
        std::lock_guard lock(m_Mutex);
        uint32_t count = 0u;
        std::erase_if(m_Cache,
                      [&](const auto& kv)
                      {
                          if (kv.second.expired())
                          {
                              count++;
                              return true;
                          }
                          return false;
                      });
        return count;
    }

    size_t LiveCount() const
    {
        std::lock_guard lock(m_Mutex);
        return std::count_if(m_Cache.begin(), m_Cache.end(),
                             [](const auto& kv) { return !kv.second.expired(); });
    }

private:
    std::unordered_map<std::string, std::weak_ptr<T>> m_Cache;
    mutable std::mutex m_Mutex;
};

} // namespace Hikari::Asset
