@echo off
setlocal

if not "%~1"=="" (
    set "PRESET=%~1"
) else (
    set "PRESET=ninja-debug-windows"
)

echo Running precommit for %PRESET%...

REM Each step below is its own top-level script with its own setlocal/
REM endlocal, so any env vars it sets (via envsetup.bat) are discarded once
REM it returns here — every step would otherwise re-run vcvars64.bat and
REM reprint its banner. Set up the toolset once here instead; envsetup.bat
REM detects the matching toolset is already active and skips re-running
REM vcvars64.bat, and that "already set up" state is inherited by each
REM `call`ed step below since it's set in this script's own scope.
if not "%PRESET%"=="msvc" (
    call "%~dp0envsetup.bat"
    if errorlevel 1 exit /b %errorlevel%
)

call build.bat "%PRESET%"
if errorlevel 1 exit /b %errorlevel%

call tests\scripts\header_check.bat "%PRESET%"
if errorlevel 1 exit /b %errorlevel%

call tests\scripts\rhi_boundary_check.bat
if errorlevel 1 exit /b %errorlevel%

call tests\scripts\namespace_check.bat
if errorlevel 1 exit /b %errorlevel%

call tests\scripts\build_tests.bat "%PRESET%"
if errorlevel 1 exit /b %errorlevel%

call tests\scripts\run_unit_tests.bat "%PRESET%"
if errorlevel 1 exit /b %errorlevel%

call tests\scripts\run_gpu_tests.bat "%PRESET%"
if errorlevel 1 exit /b %errorlevel%


call tests\scripts\run_scene_tests.bat "%PRESET%"
if errorlevel 1 exit /b %errorlevel%
call tests\scripts\format_check.bat
if errorlevel 1 exit /b %errorlevel%

echo Precommit succeeded!

endlocal
