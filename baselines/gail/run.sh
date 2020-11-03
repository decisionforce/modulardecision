#!/bin/bash
DISPLAY=:8 vglrun -d :7.2 python baselines/gail/run_carla.py --port 2000 --res 5x5 --num_trajectories 150 --num_length 600 --traj_limitation 150 --g_step 2 --render
DISPLAY=:8 vglrun -d :7.0 python baselines/gail/run_carla.py --port 2000 --res 200x100 --render 1 --num_trajectories 100 --num_length 200
