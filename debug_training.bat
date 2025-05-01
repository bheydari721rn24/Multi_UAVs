@echo off
echo Starting debug training with fewer episodes:
python run_simulation.py --mode train --num_uavs 3 --num_obstacles 3 --episodes 10 --dynamic_obstacles
