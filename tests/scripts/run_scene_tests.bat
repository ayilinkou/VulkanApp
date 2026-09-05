@echo off
setlocal

if not "%~1"=="" (
    set "PRESET=%~1"
) else (
    set "PRESET=ninja-debug-windows"
)

REM Real runs of the engine, headless, asserting on the RunReport they return.
REM Separate from the gpu label so that ctest -L gpu keeps its under-a-minute
REM meaning and a red step says which kind of thing broke. Like the gpu tests
REM they need a Vulkan ICD and skip without one — CI supplies lavapipe.
ctest --test-dir "build\%PRESET%" -L scene --output-on-failure
if errorlevel 1 exit /b %errorlevel%

endlocal
