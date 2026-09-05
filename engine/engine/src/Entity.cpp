#include "Entity.h"

Entity::Entity() : m_ID(s_NextID++)
{
    m_Name = std::string("Entity_") + std::to_string(m_ID);
    m_RootComponent.SetOwner(this);
}

/**
 * This moves a component into the SceneComponents vector. Be sure to call
 * with std::move().
 */
void Entity::AddComponent(std::unique_ptr<SceneComponent> comp)
{
    comp->SetOwner(this);
    comp->SetOwningComponent(&m_RootComponent);
    m_SceneComponents.push_back(std::move(comp));
}

/**
 * This moves a component into the LogicComponents vector. Be sure to call
 * with std::move().
 */
void Entity::AddComponent(std::unique_ptr<LogicComponent> comp)
{
    comp->SetOwner(this);
    m_LogicComponents.push_back(std::move(comp));
}
