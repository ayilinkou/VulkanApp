#include "Model.h"

#include "AssetRegistry.h"
#include "ModelData.h"

Model::Model(const std::string& path, AssetRegistry& assets)
    : m_ModelData(assets.LoadModel(path)), m_Path(path)
{
}

// Defined here rather than defaulted in the header: destroying the shared
// pointer needs ModelData complete, and the header only declares it.
Model::~Model() = default;

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
