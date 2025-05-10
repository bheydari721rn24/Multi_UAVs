@echo off
echo =================================================================
echo    AHFSI-RL Integration Training with Enhanced Obstacle Avoidance
echo =================================================================

:: Create output directories if they don't exist
mkdir .\checkpoints\ahfsi 2>nul
mkdir .\checkpoints\standard 2>nul
mkdir .\plots\training 2>nul

echo.
echo Starting AHFSI-RL Integration Training (500 episodes):
echo This will train both AHFSI-enhanced and standard models in parallel
echo for comparison. Models will be saved to ./checkpoints/ directory.
echo.
echo Press any key to start training...
pause >nul

:: Run the AHFSI-RL integration training
python ahfsi_rl_integration.py

echo.
echo Training completed! Models saved to ./checkpoints/ directory.
echo Training plots saved to ./plots/training/ directory.
echo.
echo To run demos with trained models, use: run_demo.bat
echo.
echo Press any key to exit...
pause >nul