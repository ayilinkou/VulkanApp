#pragma once

#include <cstdlib>

namespace TestEnvironment
{

/**
 * Reads a boolean-ish environment variable: set, non-empty and not "0".
 *
 * Windows-safe: MSVC deprecates std::getenv as unsafe and warnings are errors
 * here, so the platform's own accessor is used there. The same split as
 * EnableAnsiColors, for the same reason.
 */
inline bool Flag(const char* name)
{
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t length = 0u;
    if (_dupenv_s(&value, &length, name) != 0 || value == nullptr)
        return false;

    const bool bSet = value[0] != '\0' && value[0] != '0';
    std::free(value);
    return bSet;
#else
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
#endif
}

/**
 * Whether a run without a device is a failure rather than a skip.
 *
 * Set by CI, which supplies an ICD on purpose and therefore learns nothing from
 * a green run of nothing: CTest reports a skipped case as not-failed, so an
 * environment that quietly stopped providing a device looks exactly like one
 * that never had to. A developer without a GPU still gets skips.
 */
inline bool DeviceRequired()
{
    return Flag("HIKARI_TESTS_REQUIRE_DEVICE");
}

} // namespace TestEnvironment
