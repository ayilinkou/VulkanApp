#pragma once

class Entity;

class Component
{
public:
    Component() = default;
    virtual ~Component() = default;

    void SetOwner(Entity* pOwner) { m_pOwner = pOwner; }
    Entity* GetOwner() const { return m_pOwner; }

protected:
    Entity* m_pOwner = nullptr;
};
