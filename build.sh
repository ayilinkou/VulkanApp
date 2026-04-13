#!/bin/bash
cmake -B build/debug -G Ninja \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake \
	-DVCPKG_TARGET_TRIPLET=x64-linux && \
cmake --build build/debug
