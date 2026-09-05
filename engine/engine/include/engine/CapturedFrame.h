#pragma once

#include <cstdint>
#include <vector>

#include <core/Extent2D.h>

namespace Hikari::Engine
{

/**
 * The pixels of one frame, handed back instead of written.
 *
 * Always tightly packed 8-bit RGBA, whatever the present target's own format
 * was: the swizzle belongs with the code that knows the format, not with every
 * caller that encodes an image. Pixels is empty when nothing was captured.
 */
struct CapturedFrame
{
    std::vector<uint8_t> Pixels;
    Core::Extent2D Extent{};

    bool IsEmpty() const { return Pixels.empty(); }
};

} // namespace Hikari::Engine
