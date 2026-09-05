#include "Model.h"

#include "AssetRegistry.h"
#include "ModelData.h"
#include "ModelManager.h"

Model::Model(const std::string& path, AssetRegistry& assets)
    : m_ModelData(assets.LoadModel(path)), m_Path(path)
{
    ModelManager::Get()->RegisterModel(this);
}

Model::~Model()
{
    ModelManager::Get()->UnregisterModel(this);
}

std::vector<Drawable> Model::GetDrawables() const
{
    std::vector<Drawable> drawables = m_ModelData->GetDrawables();
    // TODO: not ideal, can maybe move into a GPU buffer and handle in the
    // shader
    for (Drawable& d : drawables)
    {
        d.Transform = GetAccumulatedTransform() * d.Transform;
    }
    return drawables;
}
