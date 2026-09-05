#pragma once

#include "Component.h"

/**
 * LogicComponent is a Component class which does not contain a transform.
 * Components which need a Transform should inherit from SceneComponent.
 */
class LogicComponent : public Component
{
public:
    LogicComponent(Entity* pOwner) { SetOwner(pOwner); }
};
