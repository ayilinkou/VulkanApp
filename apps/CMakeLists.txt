# Two binaries over one engine. They differ in the platform they build and the
# flags they accept; everything after that is Engine::RunApp.
foreach(app_name IN ITEMS HikariEditor HikariHeadless)
  if(app_name STREQUAL "HikariEditor")
    set(app_dir editor)
  else()
    set(app_dir headless)
  endif()

  add_executable(${app_name} ${app_dir}/main.cpp)

  # Both land beside the shaders, which are built once and read from beside
  # whichever binary is running.
  set_target_properties(${app_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                               ${HIKARI_EXE_DIR})

  target_link_libraries(
    ${app_name} PRIVATE Engine::Core Engine::Platform Engine::RHI Engine::Engine
                        Engine::Editor SanitizerShims)

  engine_set_warnings(${app_name})
  hikari_apply_build_options(${app_name})
  add_dependencies(${app_name} CompileShadersTarget)
endforeach()
