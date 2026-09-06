# Guards the RHI's public seam. Three checks, in the order the boundary is built
# up (rhi_extraction_plan.md D1, enforcement mechanism 2 in its §4):
#
#   1. A neutral header in include/rhi/ must not depend on Vulkan or VMA.
#   2. include/rhi/vulkan/, the transitional area that may expose Vulkan, holds
#      exactly the headers listed here and no others.
#   3. Outside engine/rhi/, only allowlisted sites may include that area.
#
# Checks 2 and 3 are ratchets rather than ceilings: the lists are allowed to
# shrink and an entry that stops matching is itself a failure, so neither can
# quietly outlive the code it excuses.
#
# Run with:  cmake -P cmake/RhiBoundaryCheck.cmake
# from anywhere — paths are resolved relative to this file, not the caller.
#
# Why check 1 exists alongside the HeaderSelfContainment_RHI_Neutral target: that
# target proves a neutral header compiles without linking Vulkan, but a
# dependency that also happens to sit on the default system include path is
# found regardless of what a target links, which on some distributions covers
# Vulkan. A textual check is immune to include paths. The two mechanisms fail
# independently, which is the point of having both.
#
# Why it is a CMake script rather than a shell one-liner: the .sh and .bat
# wrappers then share one implementation. Two hand-written copies of the same
# check drift, and a Windows-only or Linux-only hole in a boundary check is
# worse than no check, because it reads as covered.

cmake_minimum_required(VERSION 3.20)

get_filename_component(repo_root "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)
set(neutral_dir "${repo_root}/engine/rhi/include/rhi")

if(NOT IS_DIRECTORY "${neutral_dir}")
  message(FATAL_ERROR "rhi_boundary_check: ${neutral_dir} does not exist.")
endif()

# Types and macros first, then includes — the include patterns catch a header
# being pulled in even when nothing from it is named yet.
#
# CMake's regex flavour has no \b, so word boundaries are spelled out as "start
# of line, or a character that cannot be part of an identifier".
set(banned_patterns
    "vk::"
    "(^|[^A-Za-z0-9_])Vk[A-Z]"
    "(^|[^A-Za-z0-9_])Vma[A-Z]"
    "(^|[^A-Za-z0-9_])VMA_"
    "#[ \t]*include[ \t]*[<\"]vulkan/"
    "#[ \t]*include[ \t]*[<\"]vk_mem_alloc")

# Comments are stripped rather than matched because the neutral headers are
# expected to name Vulkan and D3D12 types in prose — recording that
# PipelineStage maps onto VkPipelineStageFlags2 and D3D12_BARRIER_SYNC is
# exactly the documentation that makes the mapping reviewable. Matching raw
# lines would make that unwritable and push the rationale out of the code.
# What is banned is a dependency, not a mention.
include("${CMAKE_CURRENT_LIST_DIR}/StripComments.cmake")

# GLOB rather than GLOB_RECURSE is load-bearing: it excludes include/rhi/vulkan/,
# the transitional area that is allowed to expose Vulkan (plan D1 and D9).
file(GLOB neutral_headers "${neutral_dir}/*.h")

if(NOT neutral_headers)
  message(FATAL_ERROR "rhi_boundary_check: no headers found in ${neutral_dir}.")
endif()

set(violations "")

foreach(header IN LISTS neutral_headers)
  file(READ "${header}" content)
  file(RELATIVE_PATH relative_header "${repo_root}" "${header}")

  set(line_number 0)
  set(in_block 0)

  # Split on newlines by hand. file(STRINGS) would turn every semicolon in the
  # source into a list separator, which would make the reported line numbers
  # meaningless.
  while(TRUE)
    string(FIND "${content}" "\n" newline_index)
    if(newline_index EQUAL -1)
      set(line "${content}")
    else()
      string(SUBSTRING "${content}" 0 ${newline_index} line)
      math(EXPR after_newline "${newline_index} + 1")
      string(SUBSTRING "${content}" ${after_newline} -1 content)
    endif()

    math(EXPR line_number "${line_number} + 1")

    strip_comments_from_line("${line}" in_block code)

    foreach(pattern IN LISTS banned_patterns)
      if(code MATCHES "${pattern}")
        list(APPEND violations "  ${relative_header}:${line_number}: ${code}")
        break()
      endif()
    endforeach()

    if(newline_index EQUAL -1)
      break()
    endif()
  endwhile()
endforeach()

if(violations)
  list(JOIN violations "\n" violation_text)
  message(
    FATAL_ERROR
      "rhi_boundary_check: neutral RHI headers must not depend on Vulkan or VMA.\n"
      "${violation_text}\n\n"
      "Backend-facing declarations belong in engine/rhi/src/vulkan/ (invisible\n"
      "outside the module) or engine/rhi/include/rhi/vulkan/ (transitional —\n"
      "exempt from this check, governed by the two below). Naming a Vulkan type\n"
      "in a comment is fine: comments are stripped before matching, so this is a\n"
      "real dependency.")
endif()

list(LENGTH neutral_headers header_count)
message(STATUS "rhi_boundary_check: ${header_count} neutral RHI header(s) free of Vulkan and VMA.")

# ---------------------------------------------------------------------------
# Check 2: the transitional area is a fixed set of headers.
#
# include/rhi/vulkan/ is the one place in the module that may expose Vulkan
# outside it (plan D1 and D9). Stage 5 ends with it frozen: a backend header
# added here rather than in src/vulkan/ has to be argued for by editing this
# list, which is the point. Everything else the backend needs is private.
# ---------------------------------------------------------------------------

set(transitional_headers
    # The escape hatch itself (D9): instance/device/queue for ImGui, and the
    # VkFormat/VkPipelineCache accessors the app's pipeline creation needs.
    "VulkanNative.h"
    # Pipeline creation stays Vulkan-side until the binding model is neutral.
    # D8 deferred this; D15 un-defers it now that D14 neutralises binding.
    # Removed by Stage 7.5 steps 6 (graphics) and 7 (compute).
    "PipelineBuilder.h"
    "ComputePipelineBuilder.h"
    # Descriptors are deliberately not abstracted in Stage 5 (D7); this is
    # isolated, not neutral. D7 expected bindless to remove it; D14 supersedes
    # that and neutralises binding directly, so it goes at Stage 7.5 steps 4-5.
    "DescriptorAllocator.h"
    # Names Vulkan objects the application still creates for itself. Shrinks as
    # those move behind the RHI; it is a template, so it cannot move to src/.
    "DebugNames.h"
    # Pure functions over surface query results. Its only production caller is
    # now SwapchainTarget, inside the module, so this could move to src/vulkan/
    # and shrink the list. It is kept here deliberately: the functions are pure
    # and device-free so that they can be unit tested, and src/vulkan/ is on a
    # PRIVATE include path, which would put them permanently out of a test's
    # reach. SwapchainUtilTests.cpp is the test that reach buys — it puts a
    # surface into states a real display cannot be asked for on demand, a zero
    # extent among them. Reconsider if it grows past choosing surface
    # parameters, or if it acquires state or a device dependency.
    "SwapchainUtil.h"
    # Begin/submit/wait for a one-shot command buffer. The remaining caller
    # records a compute dispatch, so it needs both halves: submission behind
    # IDevice (Stage 7.5 step 2) and Dispatch on ICommandList (step 11).
    "CommandListUtil.h")

set(transitional_dir "${neutral_dir}/vulkan")
file(GLOB transitional_present RELATIVE "${transitional_dir}" "${transitional_dir}/*.h")

set(unexpected "")
foreach(header IN LISTS transitional_present)
  if(NOT header IN_LIST transitional_headers)
    list(APPEND unexpected "  engine/rhi/include/rhi/vulkan/${header}")
  endif()
endforeach()

if(unexpected)
  list(JOIN unexpected "\n" unexpected_text)
  message(
    FATAL_ERROR
      "rhi_boundary_check: unexpected header in the transitional area.\n"
      "${unexpected_text}\n\n"
      "A backend header belongs in engine/rhi/src/vulkan/, where nothing\n"
      "outside the module can reach it. Put it here only if something outside\n"
      "the module must include it, and say why by adding it to\n"
      "transitional_headers in cmake/RhiBoundaryCheck.cmake.")
endif()

foreach(header IN LISTS transitional_headers)
  if(NOT header IN_LIST transitional_present)
    message(
      FATAL_ERROR
        "rhi_boundary_check: transitional_headers lists ${header}, which no\n"
        "longer exists. Delete the entry — this list is meant to shrink.")
  endif()
endforeach()

# ---------------------------------------------------------------------------
# Check 3: who outside the module may include the transitional area.
#
# The plan's target for this step was "only the ImGui glue", which Stage 5
# cannot reach: the swapchain (Stage 6), the descriptor model (D7), pipeline
# creation (D8) and dispatch recording (Stage 8) are all explicitly out of
# scope, and each of them is a reason the application still names Vulkan. So the
# rule is a ratchet instead of a ceiling — every existing use is listed with the
# work that removes it, an unlisted one fails, and an entry that stops matching
# fails too, so the list cannot quietly outlive the code it excuses.
#
# Entries are "<path>|<header>|<why it is still here>", split on "|" because a
# CMake list is already split on ";".
# ---------------------------------------------------------------------------

set(transitional_allowlist
    "engine/editor/src/VulkanUiBackend.cpp|VulkanNative.h|ImGui's backend takes instance/device/queue and a VkCommandBuffer by value (D9)"
    "engine/engine/src/Engine.cpp|PipelineBuilder.h|Graphics pipeline creation is Vulkan-side until D15 (step 6)"
    "engine/engine/src/Engine.cpp|VulkanNative.h|The frame loop still records raw draws — last use goes at step 10"
    "engine/engine/src/Engine.cpp|DebugNames.h|Names the pools, sets and sync objects the engine still owns (step 12)"
    "engine/engine/src/CloudSystem.cpp|VulkanNative.h|Raw dispatch recording needs the device — goes at step 11"
    "engine/engine/src/CloudSystem.cpp|ComputePipelineBuilder.h|Compute pipeline creation is Vulkan-side until D15 (step 7)"
    "engine/engine/src/CloudSystem.cpp|CommandListUtil.h|The noise bake is a dispatch, not a copy — needs steps 2 and 11"
    "engine/engine/src/CloudSystem.cpp|DebugNames.h|Names the bake's pipeline and descriptor set (step 12)"
    "engine/engine/src/MaterialFactory.h|DescriptorAllocator.h|Descriptors are isolated, not abstracted — bind groups replace them at step 5"
    "engine/engine/src/MaterialFactory.cpp|VulkanNative.h|Writes descriptor sets directly — bind groups replace this at step 5"
    "engine/engine/src/MaterialFactory.cpp|DebugNames.h|Names the material set layout (step 5)"
    "engine/engine/src/PBRMaterial.h|DescriptorAllocator.h|Descriptors are isolated, not abstracted — bind groups replace them at step 5"
    "engine/engine/src/PBRMaterial.cpp|VulkanNative.h|Writes descriptor sets directly — bind groups replace this at step 5"
    "engine/engine/src/PBRMaterial.cpp|DebugNames.h|Names the material descriptor set (step 5)"
    "tests/unit/rhi/SwapchainUtilTests.cpp|SwapchainUtil.h|Surface states a real display cannot be put into on demand"
    "tests/gpu/rhi/DeviceTests.cpp|VulkanNative.h|The escape hatch is what these cases assert on"
)

# Splitting by hand rather than with file(STRINGS), which would turn every
# semicolon in the source into a list separator and make line numbers useless.
function(read_lines path out_var)
  file(READ "${path}" content)
  string(REPLACE ";" "\;" content "${content}")
  string(REPLACE "\n" ";" content "${content}")
  set(${out_var} "${content}" PARENT_SCOPE)
endfunction()

file(GLOB_RECURSE scanned_files
     "${repo_root}/apps/*.h" "${repo_root}/apps/*.cpp" "${repo_root}/tests/*.h"
     "${repo_root}/tests/*.cpp" "${repo_root}/engine/*.h" "${repo_root}/engine/*.cpp")

set(unlisted "")
set(matched_entries "")

foreach(scanned IN LISTS scanned_files)
  file(RELATIVE_PATH relative_path "${repo_root}" "${scanned}")

  if(relative_path MATCHES "^engine/rhi/")
    continue()
  endif()

  read_lines("${scanned}" lines)
  set(line_number 0)

  foreach(line IN LISTS lines)
    math(EXPR line_number "${line_number} + 1")

    # Anchored at the start of the line so a commented-out include does not
    # count as a use.
    if(NOT line MATCHES "^[ \t]*#[ \t]*include[ \t]*[<\"]rhi/vulkan/([A-Za-z0-9_]+\\.h)[>\"]")
      continue()
    endif()

    set(included "${CMAKE_MATCH_1}")
    set(found FALSE)

    foreach(entry IN LISTS transitional_allowlist)
      if(entry MATCHES "^([^|]+)\\|([^|]+)\\|")
        if(CMAKE_MATCH_1 STREQUAL relative_path AND CMAKE_MATCH_2 STREQUAL included)
          set(found TRUE)
          list(APPEND matched_entries "${entry}")
          break()
        endif()
      endif()
    endforeach()

    if(NOT found)
      list(APPEND unlisted "  ${relative_path}:${line_number}: rhi/vulkan/${included}")
    endif()
  endforeach()
endforeach()

if(unlisted)
  list(JOIN unlisted "\n" unlisted_text)
  message(
    FATAL_ERROR
      "rhi_boundary_check: new use of the transitional RHI area.\n"
      "${unlisted_text}\n\n"
      "Outside engine/rhi/, rhi/vulkan/ may only be included by the sites\n"
      "listed in transitional_allowlist in cmake/RhiBoundaryCheck.cmake. Prefer\n"
      "the neutral API in rhi/. If there is genuinely no neutral way to say it\n"
      "yet, add an entry naming the work that removes it again.")
endif()

set(stale "")
foreach(entry IN LISTS transitional_allowlist)
  if(NOT entry IN_LIST matched_entries)
    string(REPLACE "|" " -> " readable "${entry}")
    list(APPEND stale "  ${readable}")
  endif()
endforeach()

if(stale)
  list(JOIN stale "\n" stale_text)
  message(
    FATAL_ERROR
      "rhi_boundary_check: transitional_allowlist has entries nothing matches.\n"
      "${stale_text}\n\n"
      "The include is gone, so delete the entry. The allowlist is a ratchet: it\n"
      "only means anything while it shrinks as the neutral API grows.")
endif()

list(LENGTH transitional_headers transitional_count)
list(LENGTH transitional_allowlist allowlist_count)
message(
  STATUS
    "rhi_boundary_check: transitional area is ${transitional_count} header(s), used from "
    "${allowlist_count} site(s) outside the module.")
