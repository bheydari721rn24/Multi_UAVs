@echo off
echo Running UAV simulation with dynamic target
echo =======================================

call .\.venv\Scripts\activate && python run_simple_demo.py --uavs 3 --obstacles 2 --ahfsi

echo.
echo Simulation complete!
echo.
pause
