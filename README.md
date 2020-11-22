# Modular Decision

## Requirements
- Ubuntu 16.04
- CARLA 0.9.5 (compiled version)
- Docker == 18.09.4, build d14af54266
- Nvidia-docker == 2.1.1
- Python == 3.6.8
- TensorFlow-GPU == 1.12

## Installation
```bash
# docker 18.09.4, build d14af54266 Ref: https://docs.docker.com/install/linux/docker-ce/ubuntu/
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
# List available docker version
$ apt-cache madison docker-ce
  
  docker-ce | 5:18.09.1~3-0~ubuntu-xenial | https://download.docker.com/linux/ubuntu  xenial/stable amd64 Packages
  docker-ce | 5:18.09.0~3-0~ubuntu-xenial | https://download.docker.com/linux/ubuntu  xenial/stable amd64 Packages
  docker-ce | 18.06.1~ce~3-0~ubuntu       | https://download.docker.com/linux/ubuntu  xenial/stable amd64 Packages
  docker-ce | 18.06.0~ce~3-0~ubuntu       | https://download.docker.com/linux/ubuntu  xenial/stable amd64 Packages
  ...
# Example $VERSION_STRING: 5:18.09.1~3-0~ubuntu-xenial
sudo apt-get install docker-ce=<VERSION_STRING> docker-ce-cli=<VERSION_STRING> containerd.io
# nvidia-docker v2.1.1: Ref: https://github.com/NVIDIA/nvidia-docker/tree/v2.1.1#ubuntu-140416041804-debian-jessiestretch
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
# Pull CARLA docker
docker pull carlasim/carla:0.9.5
# CARLA
wget -c http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.5.tar.gz
mkdir CARLA_0.9.5
tar zxvf CARLA_0.9.5.tar.gz -C CARLA_0.9.5
# Ref: https://carlachallenge.org/get-started/
git clone -b carla_challenge --single-branch https://github.com/carla-simulator/scenario_runner.git
cd scenario_runner
bash setup_environment.sh --carla-root ${PWD}/CARLA_0.9.5
source ~/.bashrc
echo "export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.5-py3.5-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI" >> ${HOME}/.bashrc or ${HOME}/.zshrc
conda env create -f environment.yml
conda activate CARLA
conda install Cython tensorflow-gpu==1.12
conda env update -f environment.yml
# (Optional) Docker without sudo
# 1. Add the `docker` group if it doesn't already exist
sudo groupadd docker
# 2. Add the connected user `$USER` to the docker group. Optionally change the username to match your preferred user.
sudo gpasswd -a $USER docker
# 3. Restart the `docker` daemon
sudo service docker restart
# If you are on Ubuntu 14.04-15.10, use `docker.io` instead:
sudo service docker.io restart
# 4. Either do a newgrp docker or log out/in to activate the changes to groups.
newgrp docker
```
## How to Run

### Render
```bash
docker kill $(docker ps -q)
python baselines/gail/run_simulators_docker.py --town 3
python scenario_runner/no_rendering_mode_095.py --show-connections --show-triggers --host 127.0.0.1
```
### Collect Demonstration Data
Run `get_data.sh` or
```bash
docker kill $(docker ps -q)
python baselines/gail/run_simulators_docker.py --town 3
python baselines/gail/gail_control.py --task generate_data --sync --num_trajectories 100 --length 200 --dim 3d --fixed_length --scenario_name Cross_Join 
```
### Train DM TRPO
```
# --model_output CL: model output, DM for the decision output, CL for the control output
docker kill $(docker ps -q)
python baselines/gail/run_simulators_docker.py --town 3
python ./baselines/gail/parallel_carla_ray.py --res 1024x768 --scenario_name OtherLeadingVehicle --speed_mode one --host 127.0.0.1 --port 2000 --algo trpo --sn --mode wp_obj --task train --num_trajectories 100 --num_length 200 --g_step 1 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 1 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 0 --init_std 1.0 --spawn_mode random --seed 0 --save_per_iter 2 --start_v 7 --sync --other_cars 6 --model_output DM --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag TRPOOvertake --scenario --rew_type
python ./baselines/gail/parallel_carla_ray.py --res 1024x768 --scenario_name Cross_Join --speed_mode one --host 127.0.0.1 --port 2000 --algo trpo --sn --mode wp_obj --task train --num_trajectories 100 --num_length 200 --g_step 1 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 1 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 0 --init_std 1.0 --spawn_mode random --seed 0 --save_per_iter 2 --start_v 7 --sync --other_cars 6 --model_output DM --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag TRPOCrossJoin --scenario --rew_type
```
### Test DM TRPO
```
# --load_model_path $MODEL_PATH
python ./baselines/gail/parallel_carla_ray.py --render --res 1024x768 --scenario_name OtherLeadingVehicle --speed_mode one --host 127.0.0.1 --port 2000 --algo trpo --sn --mode wp_obj --task evaluate --num_trajectories 100 --num_length 200 --g_step 1 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 1 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 0 --init_std 1.0 --spawn_mode random --seed 0 --save_per_iter 2 --start_v 7 --sync --other_cars 6 --model_output DM --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag TRPOOvertak --scenario --rew_type --load_model_path ../nscl-rule-based-test/checkpoint/20191115172926_trpoCLTRPOOvertak_NSCL_False_ST_TL_KL40_TR_KMidlane30_KMidlane25_CTE0.0Vel1.01nenv_1SKIP_OtherLeadingVehicle_CurSp0STD1.0_Reg0.0_DLR1e-05_Batch256_all_G0.995.G_1.D_1.G_entcoeff_0.D_entcoeff_0.001.maxkl_0.01.seed_0/model998.ckpt
```
### Rule Based method (no need to train)
```
# --rule_based
python ./baselines/gail/parallel_carla_ray.py --rule_based --scenario_name <SCENARIO_NAME> --speed_mode one --host 127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1 --port 2000_2003_2006_2009_2012_2015_2018_2021 --algo trpo --sn --mode wp_obj --task train --res 4x4 --num_trajectories 100 --num_length 200 --g_step 1 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 8 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 0 --init_std 1.0 --spawn_mode random --seed 0 --save_per_iter 2 --start_v 7 --sync --other_cars 6 --model_output DM --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag GAILCrossJoin --scenario --expert_path ./generated_expert_data_312/312_Cross_Join_21_TRPO_curriculum.pkl
```
### Train End-to-end GAIL
```
# --model_output CL: model output, DM for the decision output, CL for the control output
python ./baselines/gail/parallel_carla_ray.py --scenario_name Straight_Follow_Double --p_update --host 127.0.0.1 --port 2000 --algo trpo --sn --mode wp_obj --task train --res 1280x1024 --render --num_trajectories 100 --num_length 200 --g_step 3 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 1 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 5.5 --init_std 1.0 --spawn_mode random --seed 2 --save_per_iter 2 --start_v 9 --sync --other_cars 6 --model_output CL --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag GAILStraightFollowDouble8Actor --scenario --expert_path ./generated_expert_data_312/312_Straight_Follow_Double_18_TRPO_curriculum.pkl
```
### Test End-to-end GAIL
```
# --load_model_path $MODEL_PATH
python ./baselines/gail/parallel_carla_ray.py --scenario_name Straight_Follow_Double --p_update --host 127.0.0.1 --port 2000 --algo trpo --sn --mode wp_obj --task train --res 1280x1024 --render --num_trajectories 100 --num_length 200 --g_step 3 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 1 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 5.5 --init_std 1.0 --spawn_mode random --seed 2 --save_per_iter 2 --start_v 9 --sync --other_cars 6 --model_output CL --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag GAILStraightFollowDouble8Actor --scenario --expert_path ./generated_expert_data_312/312_Straight_Follow_Double_18_TRPO_curriculum.pkl  --load_model_path $MODEL_PATH
```
### Train DM GAIL
```bash
<SCENARIO_NAME> = Cross_Join / Ring_Join / Straight_Follow_Single / Straight_Follow_Double / Overtake / Merge_Env
--model_output DM: DeepDecision
--rule_based: Use Rule based decision module
--model_output CL: model output, DM for the decision output, CL for the control output
--scenario: if not use --scenario, it will use full map. --scenario: use `scenario runner`.
--seed: OtherLeadingVehicle GAIL seed 0 is better, others seed 2 is better
```

```bash
docker kill $(docker ps -q)
python baselines/gail/run_simulators_docker.py --town 3
# For headless systems (for machines without a display attached), 8 GPUs
python ./baselines/gail/parallel_carla_ray.py --scenario_name Cross_Join --speed_mode one --host 127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1 --port 2000_2003_2006_2009_2012_2015_2018_2021 --algo trpo --sn --mode wp_obj --task train --res 4x4 --num_trajectories 100 --num_length 200 --g_step 1 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 8 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 0 --init_std 1.0 --spawn_mode random --seed 0 --save_per_iter 2 --start_v 7 --sync --other_cars 6 --model_output DM --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag GAILCrossJoin --scenario --expert_path ./generated_expert_data_312/312_Cross_Join_21_TRPO_curriculum.pkl
# With an attached display, 8 GPUs --> render
python ./baselines/gail/parallel_carla_ray.py --render --res 1024x768 --scenario_name <SCENARIO_NAME> --speed_mode one --host 127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1_127.0.0.1 --port 2000_2003_2006_2009_2012_2015_2018_2021 --algo trpo --sn --mode wp_obj --task train --num_trajectories 100 --num_length 200 --g_step 1 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 8 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 0 --init_std 1.0 --spawn_mode random --seed 0 --save_per_iter 2 --start_v 7 --sync --other_cars 6 --model_output DM --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag GAILCrossJoin --scenario --expert_path ./generated_expert_data_312/312_Cross_Join_21_TRPO_curriculum.pkl
# With an attached display, 1 GPU -> --actor_num 1, --port 2000, --host 127.0.0.1
python ./baselines/gail/parallel_carla_ray.py --render --res 1024x768 --scenario_name OtherLeadingVehicle --speed_mode one --host 127.0.0.1 --port 2000 --algo trpo --sn --mode wp_obj --task train --num_trajectories 100 --num_length 200 --g_step 1 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 1 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 0 --init_std 1.0 --spawn_mode random --seed 0 --save_per_iter 2 --start_v 7 --sync --other_cars 6 --model_output DM --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag GAILCrossJoin --scenario --expert_path ./generated_expert_data_312/312_Cross_Join_21_TRPO_curriculum.pkl
```
### Test DM GAIL
```
# --load_model_path ./gail_ckpt/20190814122737_trpo_GAIL.DMshortL5_GAILStraightFollowSingle8ActorV31_CTE0.0Vel1.08nenv_1SKIP_Straight_Follow_SingleSTD1.0_Reg5.5_DLR1e-05_Batch256_all_G0.995.G_1.D_1.G_entcoeff_0.D_entcoeff_0.001.maxkl_0.01.seed_0/model998.ckpt
python ./baselines/gail/parallel_carla_ray.py --host 10.5.36.241 --port 2000 --algo trpo --sn --mode wp_obj --task train --res 1280x1024 --render --num_trajectories 100 --num_length 200 --g_step 1 --d_step 1 --d_model origin --train_mode all --batch_size 200 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 10 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 5.5 --init_std 1.0 --spawn_mode random --seed 0 --save_per_iter 2 --start_v 7.0 --sync --other_cars 6 --model_output DM --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0.0 --actor_nums 1 --scenario --scenario_name Straight_Follow_Single --expert_path ./generate_expert_data_312/312_Straight_Follow_Single_15_TRPO_curriculum.pkl --load_model_path ./gail_ckpt/20190814122737_trpo_GAIL.DMshortL5_GAILStraightFollowSingle8ActorV31_CTE0.0Vel1.08nenv_1SKIP_Straight_Follow_SingleSTD1.0_Reg5.5_DLR1e-05_Batch256_all_G0.995.G_1.D_1.G_entcoeff_0.D_entcoeff_0.001.maxkl_0.01.seed_0/model998.ckpt
# Or
python ./baselines/gail/parallel_carla_ray.py --render --nscl --scenario_name Cross_Join --speed_mode one --host 127.0.0.1 --port 2000 --algo trpo --sn --mode wp_obj --task train --res 1024x768 --num_trajectories 100 --num_length 200 --g_step 1 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 1 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 0 --init_std 1.0 --spawn_mode random --seed 0 --save_per_iter 2 --start_v 7 --sync --other_cars 6 --model_output DM --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag GAILCrossJoin --scenario --expert_path ./generated_expert_data_312/312_Cross_Join_21_TRPO_curriculum.pkl --load_model_path ../nscl-rule-based-test/checkpoint/20191115204502_trpo_GAIL.DMshortL5_GAILCrossJoin_NSCL_True_ST_TL_KL40_TR_KMidlane30_KMidlane25_CTE0.0Vel1.01nenv_1SKIP_Cross_Join_CurSp0STD1.0_Reg0.0_DLR1e-05_Batch256_all_G0.995.G_1.D_1.G_entcoeff_0.D_entcoeff_0.001.maxkl_0.01.seed_0/model900.ckpt
```
### Train Discriminitor Only
```
# --p_update: store_false, policy update
python ./baselines/gail/parallel_carla_ray.py --render --res 1024x768 --nscl --scenario_name OtherLeadingVehicle --p_update --speed_mode one --host 127.0.0.1 --port 2000 --algo trpo --sn --mode wp_obj --task train --num_trajectories 100 --num_length 200 --g_step 1 --d_step 1 --d_model origin --train_mode all --batch_size 256 --actor_num 1 --policy_hidden_layer 2 --dis_max 1.3 --adversary_hidden_layer 1 --d_actv tanh --dim 3d --max_iters 1000 --D_skip 1 --A_skip 1 --d_lrate 0.00001 --scene all --region 0 --init_std 1.0 --spawn_mode random --seed 0 --save_per_iter 2 --start_v 7 --sync --other_cars 6 --model_output DM --excute_mode short --curriculumn_threshold 10000000 --lanes 5 --p_pos 0. --flag GAILCrossJoin --scenario --expert_path ./generated_expert_data_312/312_Cross_Join_21_TRPO_curriculum.pkl
```

## Acknowledgement

We used parts of code from following repositories: [OpenAI/baselines](https://github.com/openai/baselines)

## Reference
**[Learning a Decision Module by Imitating Driver’s Control Behaviors
](docs/corl2020_modulardecision.pdf)**
<br />
[Junning Huang*](https://scholar.google.ca/citations?user=kaSP3zIAAAAJ&hl=en), 
[Sirui Xie*](https://scholar.google.com/citations?user=9GJn5FIAAAAJ&hl=en), 
[Jiankai Sun*](https://scholar.google.com.hk/citations?user=726MCb8AAAAJ&hl=en),
[Qiurui Ma](), 
[Chunxiao Liu](https://scholar.google.ca/citations?user=4m061tYAAAAJ&hl=en),
[Dahua Lin](https://scholar.google.ca/citations?user=GMzzRRUAAAAJ&hl=en),and
[Bolei Zhou](https://scholar.google.ca/citations?user=9D4aG8AAAAAJ&hl=en)
<br />
**In Proceedings of the Conference on Robot Learning (CoRL) 2020**
<br />
[[Paper]](https://arxiv.org/abs/1912.00191)
[[Project Page]](https://decisionforce.github.io/Modular-Decision/)

```
@InProceedings{huanglearning2020,
author={Huang, Junning and Xie, Sirui and Sun, Jiankai and Ma, Qiurui and Liu, Chunxiao and Lin, Dahua and Zhou, Bolei},
title={Learning a Decision Module by Imitating Driver’s Control Behaviors},
booktitle = {Proceedings of the Conference on Robot Learning (CoRL) 2020},
}
```
**[Neuro-Symbolic Program Search for Autonomous Driving Decision Module Design
]()**
<br />
[Jiankai Sun](https://scholar.google.com.hk/citations?user=726MCb8AAAAJ&hl=en),
[Hao Sun](), 
[Tian Han](https://scholar.google.com/citations?user=Qtvu5t4AAAAJ&hl=zh-CN),and
[Bolei Zhou](https://scholar.google.ca/citations?user=9D4aG8AAAAAJ&hl=en)
<br />
**In Proceedings of the Conference on Robot Learning (CoRL) 2020**
<br />
[[Paper]]()
[[Project Page]](https://decisionforce.github.io/modulardecision/)

```
@InProceedings{Sun_2020_corl,
author={Sun, Jiankai and Sun, Hao and Han, Tian and Zhou, Bolei}
title={Neuro-Symbolic Program Search: Towards Automatic Autonomous Driving System Design},
booktitle = {Proceedings of the Conference on Robot Learning (CoRL) 2020}
}
```

