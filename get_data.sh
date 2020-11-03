#!/bin/bash
## Cross_Join 25
docker kill $(docker ps -q)
python baselines/gail/run_simulators_docker.py --town 3
python baselines/gail/gail_control.py -a --sync --num_trajectories 200 --length 800 --dim 3d --fixed_length --scenario_name Cross_Join 
