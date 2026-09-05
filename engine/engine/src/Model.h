#pragma once

#include <memory>
#include <string>
#include <vector>

#include "Drawable.h"
#include "SceneComponent.h"

class AssetRegistry;
class ModelData;
class Material;
class Mesh;

class Model : public SceneComponent
{
public:
    /**
     * Loads through the registry it is handed rather than a global one, so a
     * model belongs to the registry that built it and to no other.
     *
     * Inert once built: it announces itself to nothing and is found by the
     * scene walk that batches it, so a model can exist with no renderer in the
     * process at all.
     */
    Model(const std::string& path, AssetRegistry& assets);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    std::vector<Drawable> GetDrawables() const;
    const std::string& GetPath() const { return m_Path; }

private:
    std::shared_ptr<ModelData> m_ModelData = nullptr;
    std::string m_Path;
};
