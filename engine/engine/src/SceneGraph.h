#pragma once

#include <memory>
#include <vector>

#include "Entity.h"
#include "Lights.h"

struct SceneGraph
{
    std::vector<DirectionalLight*> DirLights;
    std::vector<PointLight*> PointLights;
    std::vector<std::unique_ptr<Entity>> Entities;
};
