#include <catch2/catch_test_macros.hpp>

#include <string>

#include <platform/InputScript.h>

using namespace Hikari::Platform;

TEST_CASE("An empty script has no events", "[InputScript]")
{
    const InputScript script = InputScript::Parse("");

    REQUIRE(script.Events().empty());
    REQUIRE(script.LastFrame() == 0u);
}

TEST_CASE("Blank lines and comments are not commands", "[InputScript]")
{
    const InputScript script = InputScript::Parse("# a comment\n"
                                                  "\n"
                                                  "frame 3 quit   # trailing comment\n");

    REQUIRE(script.Events().size() == 1u);
    CHECK(script.Events()[0].Frame == 3u);
    CHECK(script.Events()[0].Event.Type == EventType::Quit);
}

TEST_CASE("Each command becomes the event it names", "[InputScript]")
{
    const InputScript script = InputScript::Parse("frame 5 key.down W\n"
                                                  "frame 15 key.up W\n"
                                                  "frame 20 window.resize 320x240\n"
                                                  "frame 30 screenshot\n"
                                                  "frame 40 quit\n");

    REQUIRE(script.Events().size() == 5u);

    CHECK(script.Events()[0].Event.Type == EventType::KeyDown);
    CHECK(script.Events()[0].Event.key == Key::W);
    CHECK(script.Events()[1].Event.Type == EventType::KeyUp);
    CHECK(script.Events()[2].Event.Type == EventType::Resized);
    CHECK(script.Events()[2].Event.Size.Width == 320u);
    CHECK(script.Events()[2].Event.Size.Height == 240u);
    CHECK(script.Events()[3].Event.Type == EventType::CaptureRequested);
    CHECK(script.Events()[4].Event.Type == EventType::Quit);
    CHECK(script.LastFrame() == 40u);
}

TEST_CASE("Two events on one frame keep the order they were written", "[InputScript]")
{
    const InputScript script = InputScript::Parse("frame 7 key.down A\n"
                                                  "frame 7 key.down D\n");

    REQUIRE(script.Events().size() == 2u);
    CHECK(script.Events()[0].Event.key == Key::A);
    CHECK(script.Events()[1].Event.key == Key::D);
}

TEST_CASE("A scripted event carries no native event", "[InputScript]")
{
    const InputScript script = InputScript::Parse("frame 0 quit\n");

    // Null is what tells the caller there is no window-system event behind this
    // one, which is how the UI backend knows to leave it alone.
    REQUIRE(script.Events()[0].Event.pNative == nullptr);
}

TEST_CASE("A line the format does not describe is an error", "[InputScript]")
{
    // Silently skipping an unreadable line is how a typo turns a test into a
    // test of nothing.
    CHECK_THROWS_AS(InputScript::Parse("frame 5 camera.set pos=0,2,8\n"), InputScriptError);
    CHECK_THROWS_AS(InputScript::Parse("tick 5 quit\n"), InputScriptError);
    CHECK_THROWS_AS(InputScript::Parse("frame quit\n"), InputScriptError);
    CHECK_THROWS_AS(InputScript::Parse("frame 5\n"), InputScriptError);
    CHECK_THROWS_AS(InputScript::Parse("frame 5 key.down\n"), InputScriptError);
    CHECK_THROWS_AS(InputScript::Parse("frame 5 key.down Z\n"), InputScriptError);
    CHECK_THROWS_AS(InputScript::Parse("frame 5 window.resize 320\n"), InputScriptError);
    CHECK_THROWS_AS(InputScript::Parse("frame 5 window.resize widexhigh\n"), InputScriptError);
}

TEST_CASE("Loading a file that is not there is an error rather than an empty script",
          "[InputScript]")
{
    CHECK_THROWS_AS(InputScript::Load("no/such/script.txt"), InputScriptError);
}

TEST_CASE("A script says whether it ends the run itself", "[InputScript]")
{
    // What lets a headless run go without --frames: something has to be able to
    // end a run with no window, and a quit is as good an answer as a count.
    CHECK(InputScript::Parse("frame 5 key.down W\nframe 9 quit\n").EndsRun());
    CHECK_FALSE(InputScript::Parse("frame 5 key.down W\nframe 9 screenshot\n").EndsRun());
    CHECK_FALSE(InputScript::Parse("").EndsRun());
}

TEST_CASE("Keys are named case-insensitively, movement keys and the rest alike", "[InputScript]")
{
    const InputScript script = InputScript::Parse("frame 0 key.down escape\n"
                                                  "frame 1 key.down w\n"
                                                  "frame 2 key.up F11\n");

    REQUIRE(script.Events().size() == 3u);
    CHECK(script.Events()[0].Event.key == Key::Escape);
    CHECK(script.Events()[1].Event.key == Key::W);
    CHECK(script.Events()[2].Event.key == Key::F11);
}
