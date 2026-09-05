#include <asset/ImageWriter.h>

#include <core/Log.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace Hikari::Asset
{

namespace
{
constexpr Core::LogCategory LogImageWriter("Image Writer");

/** What a capture is: four 8-bit channels, no padding between rows. */
constexpr uint32_t kBytesPerPixel = 4u;
} // namespace

bool WritePng(std::span<const uint8_t> pixels, Core::Extent2D extent, const std::string& path)
{
    const size_t expected = static_cast<size_t>(extent.Width) * extent.Height * kBytesPerPixel;
    if (pixels.size() != expected)
    {
        Core::LogMsg(Core::LogSeverity::Error, LogImageWriter,
                     "Refusing to write {}: {} pixel byte(s) for a {}x{} image, which needs {}.",
                     path, pixels.size(), extent.Width, extent.Height, expected);
        return false;
    }

    const int stride = static_cast<int>(extent.Width * kBytesPerPixel);
    const int result = stbi_write_png(path.c_str(), static_cast<int>(extent.Width),
                                      static_cast<int>(extent.Height),
                                      static_cast<int>(kBytesPerPixel), pixels.data(), stride);
    if (result == 0)
    {
        Core::LogMsg(Core::LogSeverity::Error, LogImageWriter, "Failed to write {}", path);
        return false;
    }

    return true;
}

} // namespace Hikari::Asset
