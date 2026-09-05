#pragma once

#include "glm/glm.hpp"
#include <glm/gtc/quaternion.hpp>

struct Transform
{
    glm::vec3 Position = {0.f, 0.f, 0.f};
    glm::vec3 Rotation = {0.f, 0.f, 0.f};
    glm::vec3 Scale = {1.f, 1.f, 1.f};

    glm::mat4 ToWorldMatrix() const;
    glm::mat4 ToLocalMatrix() const;
    glm::mat4 ToRotationMatrix() const;

    bool operator==(const Transform&) const = default;
};
