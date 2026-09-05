#pragma once

#include "glm/glm.hpp"

#include "Material.h"

class Mesh;

struct Drawable
{
    Mesh* pMesh = nullptr;
    Material* pMat = nullptr;
    BlendMode blendMode;
    glm::mat4 Transform = glm::mat4(1.f);

    bool operator<(const Drawable& other) const
    {
        if (blendMode != other.blendMode)
            return blendMode < other.blendMode;
        if (pMesh != other.pMesh)
            return pMesh < other.pMesh;
        return pMat < other.pMat;
    }
};
