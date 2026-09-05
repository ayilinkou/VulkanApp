#!/bin/bash
set -e

if [ -n "$1" ]; then
  PRESET="$1"
else
  OS="$(uname -s)"
  case "$OS" in
    Linux)  PRESET="ninja-debug-linux" ;;
    *)
      echo "Unsupported OS: $OS" >&2
      exit 1
      ;;
  esac
fi

# --borderless is what fixes the screenshot's size, not --resolution. A window
# size is a request the window system may refuse, and a tiling compositor
# (Hyprland, sway, i3) always does — it puts the window in whatever tile the
# layout says, so the captured frame comes out at the tile's size and differs
# between machines and between layouts. Covering the display is honoured, so it
# gives the same extent on every run.
#
# --resolution is still passed: it is what a non-tiling window system uses, and
# it is the size the window is created at before the mode change, so a
# compositor that refuses fullscreen degrades to the right size rather than to
# three quarters of the display.
#
# Both only pin the extent to *this* display's. A capture that does not depend
# on the display at all needs an offscreen render target (Part IV steps 38-39).
#
# --no-ui is what makes the capture about the scene. The editor panel is a fifth
# of the frame, and it is not reproducible on principle: nothing warps the
# cursor at startup, so the panel carries a hover highlight on whichever widget
# the mouse was last over. It has been stable in practice only because the mouse
# did not move between runs. ImGui still initialises and its pass still records,
# so the counters in the report are unaffected by the flag.
./build/$PRESET/HikariEditor --report --screenshot --frames --fixed-dt --scene --camera-preset 1 \
    --resolution 1920x1080 --borderless --no-ui
