#pragma once

#include "Component.h"
#include "Transform.h"

/**
 * SceneComponent is a Component class which contains a Transform. Components
 * which do not need a Transform should inherit from LogicComponent.
 */
class SceneComponent : public Component
{
public:
    SceneComponent() = default;
    virtual ~SceneComponent() = default;

    void SetOwningComponent(SceneComponent* pComp) { m_pOwningComp = pComp; }

    Component* GetOwningComponent() const { return m_pOwningComp; }
    Transform& GetTransform() { return m_Transform; }

    glm::mat4 GetAccumulatedTransform() const;

protected:
    Transform m_Transform;
    SceneComponent* m_pOwningComp = nullptr;
};
