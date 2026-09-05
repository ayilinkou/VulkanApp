#include "SceneComponent.h"

glm::mat4 SceneComponent::GetAccumulatedTransform() const
{
    // if this component is the root component
    if (m_pOwningComp == nullptr)
    {
        // This component is the root component. Its translation should
        // not be affected by scale and so will use ToWorldMatrix()
        return m_Transform.ToWorldMatrix();
    }

    // We want the transform to be affected by the parent's scale so
    // ToLocalMatrix() is used here.
    return m_pOwningComp->GetAccumulatedTransform() * m_Transform.ToLocalMatrix();
}
