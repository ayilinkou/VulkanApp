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

echo "Running precommit for $PRESET..."

./build.sh "$PRESET" && \
tests/scripts/header_check.sh "$PRESET" && \
tests/scripts/rhi_boundary_check.sh && \
tests/scripts/namespace_check.sh && \
tests/scripts/build_tests.sh "$PRESET" && \
tests/scripts/run_unit_tests.sh "$PRESET" && \
tests/scripts/run_gpu_tests.sh "$PRESET" && \
tests/scripts/run_scene_tests.sh "$PRESET" && \
tests/scripts/format_check.sh && \

echo "Precommit succeeded!"
