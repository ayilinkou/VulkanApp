#pragma once

#include "glm/glm.hpp"

#include "SceneComponent.h"

class Camera : public SceneComponent
{
public:
    Camera();

    glm::vec3 GetPosition() const { return m_Transform.Position; }
    glm::mat4 GetViewMatrix() const { return m_View; }
    glm::mat4 GetProjMatrix() const { return m_Proj; }
    float GetMoveSpeed() const { return m_MoveSpeed; }
    glm::vec3 GetForwardVector() const { return m_ForwardVector; }
    glm::vec3 GetRightVector() const { return m_RightVector; }
    float GetNearPlane() const { return m_NearPlane; }
    float GetFarPlane() const { return m_FarPlane; }
    float GetFOV() const { return m_FOV; }

    void Tick();

    void Rotate(float x, float y);
    void SetProjection(const glm::mat4& newProj) { m_Proj = newProj; }

    /** FOV in degrees */
    void SetProjection(float fov, float aspect, float near, float far);

private:
    void CalcViewMatrix();

private:
    glm::mat4 m_View = glm::mat4(1.f);
    glm::mat4 m_Proj;
    glm::mat4 m_RotationMatrix = glm::mat4(1.f);
    float m_FOV = 90.f;
    float m_NearPlane = 0.1f;
    float m_FarPlane = 10000.f;

    glm::vec3 m_ForwardVector{0.f, 0.f, -1.f};
    glm::vec3 m_UpVector = {0.f, 1.f, 0.f};
    glm::vec3 m_RightVector = {1.f, 0.f, 0.f};

    float m_MoveSpeed = 5.f;
    float m_LookSens = 0.1f;
};
