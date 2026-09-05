#pragma once

#include <chrono>

namespace Hikari::Core
{

/**
 * Where the simulation gets its time.
 *
 * Only the simulation. The run report's timings measure how long things
 * actually took and read the steady clock directly — a measurement that a
 * fixed timestep silently rewrote would be worthless. This is the clock that
 * decides how far the world moves in a frame, which is a different question
 * from how long the frame took.
 */
class IClock
{
public:
    virtual ~IClock() = default;

    /** Advances to a new frame and returns the seconds since the previous one. */
    virtual float Tick() = 0;

    /**
     * Seconds since the clock started, as of the last Tick rather than as of
     * now: two things reading it within a frame must agree on what time it is.
     */
    virtual float Elapsed() const = 0;
};

/** Wall-clock time, for a run whose pace is whatever the machine manages. */
class RealClock final : public IClock
{
public:
    RealClock() : m_Start(std::chrono::steady_clock::now()), m_Last(m_Start) {}

    float Tick() override
    {
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        const float delta = Seconds(now - m_Last);
        m_Last = now;
        m_Elapsed = Seconds(now - m_Start);
        return delta;
    }

    float Elapsed() const override { return m_Elapsed; }

private:
    static float Seconds(std::chrono::steady_clock::duration duration)
    {
        return std::chrono::duration<float, std::chrono::seconds::period>(duration).count();
    }

    std::chrono::steady_clock::time_point m_Start;
    std::chrono::steady_clock::time_point m_Last;
    float m_Elapsed = 0.f;
};

/**
 * The same step every frame, whatever the machine did.
 *
 * What makes a run repeatable: the world advances by frame count rather than
 * by elapsed time, so the same frame number is the same simulated moment on a
 * fast machine, a slow one, and a headless one that renders as fast as it can.
 */
class FixedStepClock final : public IClock
{
public:
    /** Defaults to 1/60s, which is what --fixed-dt has always meant. */
    explicit FixedStepClock(float stepSeconds = 1.f / 60.f) : m_Step(stepSeconds) {}

    float Tick() override
    {
        m_Elapsed += m_Step;
        return m_Step;
    }

    float Elapsed() const override { return m_Elapsed; }

private:
    float m_Step;
    float m_Elapsed = 0.f;
};

} // namespace Hikari::Core
