@echo off
echo =================================================================
echo    AHFSI-RL Integration Demo with Enhanced Obstacle Avoidance
echo =================================================================

:: Create output directories if they don't exist
mkdir .\plots 2>nul

echo.
echo Available Demo Options:
echo   1. AHFSI Model with Military Visualization
echo   2. Standard Model with Military Visualization

:menu
echo.
set /p option="Enter option (1-2): "

if "%option%"=="1" (
    echo.
    echo Running AHFSI Model with Military Visualization...
    echo Using the proper military visualization system...
    call .\.venv\Scripts\activate && python run_simple_demo.py --uavs 3 --obstacles 2 --ahfsi
) else if "%option%"=="2" (
    echo.
    echo Running Standard Model with Military Visualization...
    echo Using the proper military visualization system...
    call .\.venv\Scripts\activate && python run_simple_demo.py --uavs 3 --obstacles 2 --no-ahfsi
) else (
    echo.
    echo Invalid option. Please try again.
    goto menu
)

echo.
echo Demo completed! Any animations have been saved to ./plots/ directory.
echo.
echo Press any key to return to the menu, or Ctrl+C to exit...
pause >nul
goto menu