#include "Material.h"

#include "assimp/material.h"

Material::Material(aiMaterial* pMat)
    : m_Name(pMat->GetName().C_Str()), m_BlendMode(DetectBlendMode(pMat))
{
}

BlendMode Material::DetectBlendMode(aiMaterial* pMat)
{
    float opacity = 1.f;
    if (pMat->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS && opacity < 1.f)
        return BlendMode::Transparent;

    aiColor4D baseColor;
    if (pMat->Get(AI_MATKEY_BASE_COLOR, baseColor) == AI_SUCCESS && baseColor.a < 1.f)
        return BlendMode::Transparent;

    return BlendMode::Opaque;
}
