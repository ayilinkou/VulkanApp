# The asset layer. Small for now: the cache, and the image encoding that used to
# sit in the application. The registry that owns them both still names types
# that live in the Engine module, and follows them here when Stage 8 splits it.
engine_module(Asset SOURCES src/ImageWriter.cpp LINK_LIBRARIES Engine::Core)

# SYSTEM, so that stb's own warnings are not this module's problem: the
# implementation is compiled here and it does not build clean under -Wextra.
target_include_directories(Asset SYSTEM PRIVATE ${Stb_INCLUDE_DIR})
