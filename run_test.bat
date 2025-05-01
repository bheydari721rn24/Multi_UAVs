@echo off

:: Create output directories if they don't exist
mkdir .\output\dynamic 2>nul
mkdir .\output\static 2>nul

echo Testing models with random seeds:
python run_simulation.py --mode test --load_path ./saved_models/dynamic --num_uavs 3 --num_obstacles 3 --episodes 1 --output ./output/dynamic --dynamic_obstacles --num_runs 1 --logging
python run_simulation.py --mode test --load_path ./saved_models/static --num_uavs 3 --num_obstacles 3 --episodes 1 --output ./output/static --num_runs 1 --logging

echo.
echo Testing models with fixed seeds (reproducible results):
python run_simulation.py --mode test --load_path ./saved_models/dynamic --num_uavs 3 --num_obstacles 3 --episodes 1 --output ./output/dynamic --dynamic_obstacles --seed 12345 --logging
python run_simulation.py --mode test --load_path ./saved_models/static --num_uavs 3 --num_obstacles 3 --episodes 1 --output ./output/static --seed 12345 --logging

echo.
echo Testing completed! Results saved to ./output/ directory.
echo Press any key to exit...
pause >nul
