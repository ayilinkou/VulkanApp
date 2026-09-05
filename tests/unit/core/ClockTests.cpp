#include <catch2/catch_test_macros.hpp>

#include <chrono>

#include <core/Clock.h>

using namespace Hikari::Core;

namespace
{
/** Spins until the steady clock has actually moved, without sleeping the suite. */
void WaitForRealTimeToAdvance()
{
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() == start)
    {
    }
}
} // namespace

TEST_CASE("A fixed step returns the same delta every frame", "[Clock]")
{
    FixedStepClock clock;

    REQUIRE(clock.Tick() == 1.f / 60.f);
    REQUIRE(clock.Tick() == 1.f / 60.f);
    REQUIRE(clock.Tick() == 1.f / 60.f);
}

TEST_CASE("A fixed step accumulates elapsed time by frame count", "[Clock]")
{
    FixedStepClock clock(0.5f);

    // Nothing has happened yet, so no time has passed — a frame's worth of time
    // arrives with the frame, not before it.
    REQUIRE(clock.Elapsed() == 0.f);

    clock.Tick();
    REQUIRE(clock.Elapsed() == 0.5f);

    clock.Tick();
    REQUIRE(clock.Elapsed() == 1.f);
}

TEST_CASE("Two fixed-step clocks agree frame for frame", "[Clock]")
{
    FixedStepClock first;
    FixedStepClock second;

    // The property the whole step exists for: the same frame number is the same
    // simulated moment, whatever the machine did between them.
    for (int frame = 0; frame < 100; ++frame)
    {
        first.Tick();
        WaitForRealTimeToAdvance();
        second.Tick();
    }

    REQUIRE(first.Elapsed() == second.Elapsed());
}

TEST_CASE("A real clock reports no time before its first tick", "[Clock]")
{
    RealClock clock;

    WaitForRealTimeToAdvance();

    REQUIRE(clock.Elapsed() == 0.f);
}

TEST_CASE("A real clock's elapsed time only moves on a tick", "[Clock]")
{
    RealClock clock;

    clock.Tick();
    const float first = clock.Elapsed();

    // Two readers within one frame have to agree on what time it is, so real
    // time passing between them must not show up.
    WaitForRealTimeToAdvance();
    REQUIRE(clock.Elapsed() == first);

    clock.Tick();
    REQUIRE(clock.Elapsed() > first);
}

TEST_CASE("A real clock's deltas are never negative", "[Clock]")
{
    RealClock clock;

    for (int frame = 0; frame < 100; ++frame)
    {
        REQUIRE(clock.Tick() >= 0.f);
    }
}
