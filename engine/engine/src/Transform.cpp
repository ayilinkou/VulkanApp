#include "Transform.h"

/**
 * Order: Scale -> Rotation -> Translation. Translation is NOT affected by
 * scale.
 */
glm::mat4 Transform::ToWorldMatrix() const
{
    glm::mat4 mat(1.f);
    mat = glm::scale(mat, Scale);
    mat = glm::mat4_cast(glm::quat(glm::radians(Rotation))) * mat;
    mat[3] = glm::vec4(Position, 1.f);
    return mat;
};

/**
 * Order: Scale -> Rotation -> Translation. Translation IS affected by
 * scale.
 */
glm::mat4 Transform::ToLocalMatrix() const
{
    glm::mat4 mat(1.f);
    mat = glm::scale(mat, Scale);
    mat = glm::mat4_cast(glm::quat(glm::radians(Rotation))) * mat;
    mat = glm::translate(glm::mat4(1.f), Position) * mat;
    return mat;
};

glm::mat4 Transform::ToRotationMatrix() const
{
    return glm::mat4_cast(glm::quat(glm::radians(Rotation)));
}
