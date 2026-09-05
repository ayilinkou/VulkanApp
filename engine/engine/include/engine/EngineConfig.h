#pragma once

#include <cstdint>

#include <glm/vec3.hpp>

namespace Hikari::Engine
{

/**
 * What the engine is built with, as against what a run does — that is RunSpec.
 *
 * Everything here sizes something once, at construction: the frames in flight
 * decide how many of every per-frame resource exist, and the instance capacity
 * is the first allocation of a buffer that grows on demand. Changing one of
 * them mid-run would mean rebuilding what it sized, which is why they are a
 * construction-time input rather than part of the run description.
 */
struct EngineConfig
{
    /**
     * How many frames the CPU may work on at once. One makes every frame wait
     * for the previous one to finish, which is the path the per-frame resources
     * are least often exercised on.
     */
    uint32_t FramesInFlight = 2u;

    /**
     * Instances every frame's buffer starts with. A starting size, not a
     * ceiling — the buffer grows when a frame needs more.
     */
    uint32_t InitialInstanceCapacity = 1024u;

    /** The opaque pass's clear colour, and the sky colour the shaders read. */
    glm::vec3 SkyColor = {0.4f, 0.8f, 1.f};
};

} // namespace Hikari::Engine
