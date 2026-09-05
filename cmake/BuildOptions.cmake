# hikari_apply_build_options(<target>)
#
# The per-configuration compile and link options the application used to carry
# alone. Applied to the targets that compile the bulk of the code — the Engine
# module and the two apps — rather than to everything, so that a leaf module
# does not pay for a linker choice it cannot benefit from.
function(hikari_apply_build_options target)
  if(MSVC)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
      # MSBuild batches TUs into one cl invocation, so /MP is what parallelises.
      target_compile_options(${target} PRIVATE /MP)
    else()
      # Ninja parallelises itself; /Z7 embeds debug info in each .obj and avoids
      # serialising all TUs through mspdbsrv writing a shared PDB.
      set_target_properties(${target} PROPERTIES MSVC_DEBUG_INFORMATION_FORMAT
                                                 "$<$<CONFIG:Debug>:Embedded>")
    endif()

    target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:/Od>
                                             $<$<CONFIG:Release>:/O2>)

    get_target_property(target_type ${target} TYPE)
    if(target_type STREQUAL "EXECUTABLE")
      target_link_options(${target} PRIVATE $<$<CONFIG:Debug>:/DEBUG>
                          $<$<CONFIG:Release>:/DEBUG>)
    endif()
  else()
    target_compile_options(
      ${target}
      PRIVATE $<$<CONFIG:Debug>:
              -O0
              -g
              >
              $<$<CONFIG:Release>:
              -O3
              >)

    # lld links noticeably faster than bfd, which matters most for the ASan/UBSan
    # build (large instrumented runtime + debug info to link).
    get_target_property(target_type ${target} TYPE)
    if(target_type STREQUAL "EXECUTABLE")
      include(CheckLinkerFlag)
      check_linker_flag(CXX "-fuse-ld=lld" LLD_LINKER_AVAILABLE)
      if(LLD_LINKER_AVAILABLE)
        target_link_options(${target} PRIVATE -fuse-ld=lld)
      endif()
    endif()
  endif()
endfunction()
