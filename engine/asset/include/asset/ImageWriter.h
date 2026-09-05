#pragma once

#include <cstdint>
#include <span>
#include <string>

#include <core/Extent2D.h>

namespace Hikari::Asset
{

/**
 * Writes tightly packed 8-bit RGBA pixels out as a PNG.
 *
 * Here because this module already owns the image library for decoding, and a
 * second copy of it linked elsewhere is how two halves of a program come to
 * disagree about a format. Returns false and logs on failure.
 */
bool WritePng(std::span<const uint8_t> pixels, Core::Extent2D extent, const std::string& path);

} // namespace Hikari::Asset
