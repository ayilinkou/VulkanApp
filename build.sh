#!/bin/bash
cmake -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug && \
cmake --build build/debug
