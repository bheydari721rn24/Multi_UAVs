@echo off
echo Starting new training (without loading previous models):
python run_simulation.py --mode train --save_path ./saved_models/dynamic --num_uavs 3 --num_obstacles 3 --episodes 30000 --dynamic_obstacles --num_runs 1
python run_simulation.py --mode train --save_path ./saved_models/static --num_uavs 3 --num_obstacles 3 --episodes 30000 --num_runs 1

echo.
echo Starting continued training (loading previous models):
python run_simulation.py --mode train --save_path ./saved_models/dynamic --load_path ./saved_models/dynamic --num_uavs 3 --num_obstacles 3 --episodes 30000 --dynamic_obstacles --seed 12345
python run_simulation.py --mode train --save_path ./saved_models/static --load_path ./saved_models/static --num_uavs 3 --num_obstacles 3 --episodes 30000 --seed 12345

echo.
echo Training completed! Press any key to exit...
pause >nul