'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm

import numpy as np
import gym

import sys
sys.path.append("../../")
import glob, os

baseline_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, baseline_dir)
#
## add carla config here, put carla directory as the parent directory of IDM
#carla_simulator_path = os.path.dirname(os.path.dirname(baseline_dir))
#carla_simulator_path += '/PythonAPI/carla-0.9.1-py3.5-linux-x86_64.egg'
carla_simulator_path = '/home/SENSETIME/maqiurui/reinforce/carla/carla_0.9.5/PythonAPI/carla-0.9.5-py3.5-linux-x86_64.egg'
try: 
    sys.path.append(carla_simulator_path)
    sys.path.append(baseline_dir+'/../PythonAPI/carla-0.9.5-py3.5-linux-x86_64.egg')
except IndexError:
    pass

from baselines.gail.gail_control import Carla, CarlaDM, carla, World, RoadOption
from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.dataset.mujoco_dset import Carla_Dset, Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier
from baselines.logger import TensorBoardOutputFormat
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.dummy_vec_env import DummyCarlaEnv
from baselines.common.vec_env.vec_monitor import VecMonitor
from copy import copy

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--sync', help='whether to sync server to client',action='store_true')
    parser.add_argument('--env_id', help='environment ID', default='Carla-Motion')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='/mnt/lustre/huangjunning/CARLA/carla_094_compiled/IDM/baselines/log/easy.pkl')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample', 'bc_evaluate'], default='train')
    # for evaluatation
    #boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    #boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--stochastic_policy',action='store_true', help='use stochastic/deterministic policy to evaluate')
    parser.add_argument('--save_sample',action='store_true',help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=10, help='useless in carla')
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=1)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--policy_hidden_layer', type=int, default=2)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_layer', type=int, default=1)
    parser.add_argument('--d_actv', type=str, default="tanh", help='Activation for discriminator, default is tanh')
    parser.add_argument('--sn', action='store_false', help='Spectral normalization on Discriminator')
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    parser.add_argument('--d_lrate', type=float, default=0.00001)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=20)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=0)
    # Behavior Cloning
    #boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--pretrained', action='store_true',help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=2e4)
    # Carla settings
    # parser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('--host',default='127.0.0.1',type=str, help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('--port',default='2000',type=str,help='TCP port to listen to (default: 2000)')
    parser.add_argument('--A_skip',default=1,type=int,help="skip frame number")
    parser.add_argument('--D_skip',default=1,type=int,help="skip frame number")
    parser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    parser.add_argument('--render', action='store_true', help='Render botton')
    parser.add_argument('--birdview',action='store_true',help='use bird view')
    parser.add_argument('--draw',action='store_true',help='draw waypoints in the hud')
    parser.add_argument('--num_trajectories', metavar='T', default=200, type=int, help='num of trajectories')
    parser.add_argument('--num_length', metavar='L', default=800, type=int, help='length of one trajectory')
    parser.add_argument('--batch_size', default=2048, type=int, help='timesteps per batch')
    parser.add_argument('--search', action='store_true', help='search for nearest expert path for training')
    parser.add_argument('--stack', default=1, type=int, help='stack frames')
    parser.add_argument('--search_mode', default='traj', type=str, help='search mode, default is nearest trajectory')
    parser.add_argument('--scene', default='all', type=str, choices=['all', 'straight', 'curve'], help='training scene')
    parser.add_argument('--mode', default='wp_obj', type=str, choices=['all', 'wp', 'wp_obj'], help='visible ranges for observation')
    parser.add_argument('--train_mode', default='all', type=str, choices=['all', 'steer'], help='choose which action to train')
    parser.add_argument('--feature', default='lane_car', type=str, choices=['wp_car', 'lane_car'], help='features')
    parser.add_argument('--d_model', default='origin', type=str, choices=['origin', "separate"], help='model for discriminator')
    parser.add_argument('--p_update', action='store_false', help='policy update')
    parser.add_argument('--rew_type', action='store_true', help='true reward to update')
    parser.add_argument('--detect', action='store_true', help='whether to detect right or wrong')
    parser.add_argument('--dim', default='3d', type=str, choices=['2d', '3d'], help='observations dimension')
    parser.add_argument('--region', default=0.2, type=float, help='region for early reset')
    parser.add_argument('--resampling', default=0, type=int, choices=[0, 4, 8, 12], help='resampling for increasing observation diversity')
    parser.add_argument('--gamma', default=0.995, type=float, help='gamma discounting factor')
    parser.add_argument('--dis_max', default=1.3, type=float, help='longest distance for lane sampling')
    parser.add_argument('--r_norm', action='store_true', help='reward normalization')
    parser.add_argument('--init_std', default=1., type=float, help='initial std')
    parser.add_argument('--carla_path', default='/data/carla_091_compiled/', type=str, help='relative path of the folder of carlaUE4.sh')
    parser.add_argument('--spawn_mode', default='random', type=str, choices=['fixed', 'random'], help='spawn mode')
    parser.add_argument('--pretrain_std',action='store_true',help='pretrain std')
    parser.add_argument('--still_std',action='store_true',help='hold std still during train')
    parser.add_argument('--start_v',default=6.4,type=float,help='start velocity')
    parser.add_argument('--max_iters', default=1000,type=int,help='max iters')
    parser.add_argument('--sigma_pos', default=0.6,type=float,help='sigma for track position reward, model as gaussian distribution')
    parser.add_argument('--sigma_vel_upper', default=3,type=float,help='sigma for velocity reward')
    parser.add_argument('--sigma_vel_lower', default=0.6,type=float,help='sigma for velocity reward')
    parser.add_argument('--sigma_ang', default=0.4,type=float,help='sigma for angle reward')
    parser.add_argument('--curriculumn_threshold', default=5,type=float,help='sigma for angle reward')
    parser.add_argument('--other_cars',default=30,type=int,help='the number of other cars')
    parser.add_argument('--model_output', default='DM', type=str, choices=['DM', 'CL'], help='model output, DM for the decision output, CL for the control output')
    parser.add_argument('--excute_mode', default='short', type=str, choices=['short', 'long'], help='excute mode for the decision model')
    parser.add_argument('--lanes', default=3, type=int, choices=[1, 3, 5, 7], help='total lane numbers')
    parser.add_argument('--overtake_curriculum', action='store_true', help='overtake curriculum botton')
    parser.add_argument('--flag', default='default', type=str, help='excute mode for the decision model')
    parser.add_argument('--scenario_name', default='OtherLeadingVehicle', type=str, help='excute mode for the decision model')
    parser.add_argument('--p_pos', default=0., type=float, choices=[0., 1.0, 0.1, 0.01, 0.001], help='Propotion parameter for the cte error in the PID controller')
    parser.add_argument('--p_vel', default=1.0, type=float, choices=[1.0, 0.8, 0.6, 0.4, 0.2], help='Propotion parameter for the longitudinal control in the PID controller')
    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    return args


def get_task_name(args):
    import time
    time_pre = time.strftime("%Y%m%d%H%M%S", time.localtime())
    if args.rew_type:
        task_name = time_pre+'_'+args.algo
    else:
        task_name = time_pre+'_'+args.algo+ "_GAIL."
    if args.model_output == 'DM':
        task_name = task_name+args.model_output+args.excute_mode+'L' +str(args.lanes)+'_'
    elif args.model_output == 'CL':
        task_name = task_name + args.model_output
    if args.flag != 'default':
        task_name += '%s_' % args.flag
    task_name += 'CTE%sVel%s' % (str(args.p_pos), str(args.p_vel))
    task_name += '%snenv' % len(args.host.split('_'))
    task_name += '_%sSKIP' % str(args.D_skip)
    task_name += '_%s' % args.scenario_name
    if args.r_norm:
        task_name += 'Norm'
    if args.pretrained:
        task_name += "PreT%s." % args.BC_max_iter
    if args.still_std:
        task_name += "_still_std_"
    if args.search:
        task_name += "_Search%s_" % args.search_mode
    if args.sn:
        task_name += "SN_"
    if not args.p_update:
        task_name += 'NoP'
    #task_name += 'RSig_%s_%s_%s_%s' % (args.sigma_pos, args.sigma_vel_upper, args.sigma_vel_lower, args.sigma_ang)
    task_name += 'STD%s_'% args.init_std
    PLayer = args.policy_hidden_layer * 2 + 1
    #task_name += 'D%s' % args.d_model
    task_name += 'Reg%s_' % args.region
    task_name += 'MaxD%s_' % args.log_dis_max
    if args.search:
        task_name += 'Lim%s_' % args.traj_limitation
    task_name += 'GSize%s_' % args.policy_hidden_size
    task_name += 'PLay%s_' % PLayer
    task_name += 'DSize%s_' % args.adversary_hidden_size
    if args.d_model == "separate":
        adversary_hidden_layer = args.adversary_hidden_layer - 1
        DLayer = adversary_hidden_layer + 3
    else:
        adversary_hidden_layer = args.adversary_hidden_layer 
        DLayer = adversary_hidden_layer + 2
    if args.resampling != 0:
        task_name += 'ReS_%s' % args.resampling
    task_name += 'DLay%s_' % DLayer
    task_name += 'DAtv%s_' % args.d_actv
    task_name += 'DLR%s_' % args.d_lrate
    task_name += 'Batch%s_' % args.batch_size
    
    #task_name += '%s_' %args.scene
    #task_name += 'G%s'%args.gamma

    #task_name = task_name + ".G_" + str(args.g_step) + ".D_" + str(args.d_step) + \
        #".G_entcoeff_" + str(args.policy_entcoeff) + ".D_entcoeff_" + str(args.adversary_entcoeff) + \
        #".maxkl_" + str(args.max_kl)
    task_name += ".seed_" + str(args.seed)
    return task_name

def recover_args(args):
    sigmas = {}
    sigmas.update({'sigma_pos': args.sigma_pos})
    sigmas.update({'sigma_vel_upper': args.sigma_vel_upper})
    sigmas.update({'sigma_vel_lower': args.sigma_vel_lower})
    sigmas.update({'sigma_ang': args.sigma_ang})
    args.sigmas = sigmas
    return args

def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    args = recover_args(args)

    # start carla clients
    envs = []
    host = [str(x) for x in args.host.split('_') if x!='']
    port = [int(x) for x in args.port.split('_') if x!='']
    print(host,port)
    assert len(host)==len(port), 'number of hosts and ports should match'
    logger.p_pos = args.p_pos
    logger.p_vel = args.p_vel

    for i in range(len(host)):
        client = carla.Client(host[i], port[i])
        client.set_timeout(50.0)

        carla_world = client.get_world()
        assert not(args.A_skip != 1 and args.D_skip != 1)
        world = World(carla_world, args.sync, args.sigmas, camera=args.render, A_skip=args.A_skip, mode=args.mode, feature=args.feature, dim=args.dim, dis_max=args.dis_max, spawn_mode=args.spawn_mode, render=args.render, width=args.width, height=args.height,other_cars=args.other_cars,curriculumn_threshold=args.curriculumn_threshold, max_lanes=args.lanes, scenario_name=args.scenario_name)
        args.log_dis_max = int(args.dis_max**16)

        if args.task == "evaluate":
            test = True
        else:
            test = False

        if args.model_output == 'DM':
            env = CarlaDM(world, args.num_length, args.stack, args.train_mode, test, args.region, start_v=args.start_v, excute_mode=args.excute_mode, D_skip=args.D_skip, overtake_curriculum=args.overtake_curriculum)
        elif args.model_output == 'CL':
            env = Carla(world, args.num_length, args.stack, args.train_mode, test, args.region, start_v=args.start_v)
        gym.logger.setLevel(logging.WARN)
         
        env.seed(args.seed)
        envs.append(env)
    # @Junning: menv wrapper
    env = DummyCarlaEnv(envs)
    env = VecFrameStack(env, args.stack)
    env = VecMonitor(env)
    
    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=args.policy_hidden_layer)
    
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    logger.init_std = args.init_std

    if args.task == 'train':
        datashape = env.observation_space.shape[0]
        dataset = None
        if not args.rew_type or args.pretrained: # Using TRPO
            if args.env_id == "Carla-Motion":
                dataset = Carla_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation, num_trajectories=args.num_trajectories, num_length=args.num_length, data_shape=datashape, search=args.search, mode=args.mode, feature=args.feature, train_mode=args.train_mode, dim=args.dim, resampling=args.resampling)
            else:
                dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        logger.ckpt_dir_name = args.checkpoint_dir
        logger.configure(args.checkpoint_dir, format_strs=['stdout', 'json', 'log', 'tensorboard'])
        logger.num_length = args.num_length
        logger.log('Spawn mode: '+str(args.spawn_mode))
        logger.log('Trajs: '+str(args.num_trajectories))
        logger.log('Length: '+str(args.num_length))
        logger.log('Stack: '+str(args.stack))
        logger.log('Mode: '+str(args.mode))
        logger.log('Train mode: '+str(args.train_mode))
        logger.log('Feature: '+str(args.feature))
        logger.log('dim: '+str(args.dim))
        logger.log('Pretrain_std: '+str(args.pretrain_std))
        logger.log('Expected start velocity: '+str(args.start_v))
        logger.log('Decision skip: '+str(args.D_skip))
        logger.log('Action skip: '+str(args.A_skip))
        logger.log('Overtake curriculum: '+ str(args.overtake_curriculum))
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff, adversary_hidden_layer=args.adversary_hidden_layer, sn=args.sn, d_actv=args.d_actv, lr_rate=args.d_lrate, model=args.d_model, train_mode=args.train_mode, dim=args.dim)
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              task_name,
              args.load_model_path,
              args.batch_size,
              args.search,
              args.search_mode,
              args.scene,
              args.p_update,
              args.rew_type,
              args.train_mode,
              args.gamma,
              args.r_norm,
              args.pretrain_std,
              args.still_std,
              args.max_iters,
              args.model_output
              )
    elif args.task == 'evaluate':
        datashape = env.observation_space.shape[0]
        logger.ckpt_dir_name = args.checkpoint_dir
        logger.configure(args.checkpoint_dir)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff, adversary_hidden_layer=args.adversary_hidden_layer, 
                                            sn=args.sn, d_actv=args.d_actv, lr_rate=args.d_lrate, model=args.d_model, detect=args.detect, train_mode=args.train_mode,
                                            model_output=args.model_output)
        if not args.rew_type or args.pretrained:
            dataset = Carla_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation, num_trajectories=args.num_trajectories, num_length=args.num_length, data_shape=datashape, search=args.search, mode=args.mode, feature=args.feature)
        else:
            dataset = None
        # if algorithm is TRPO, env = Carla class
        # elif algorithm is PPO, env = Dummy vector class
        # TODO: runner ppo version
        if args.algo == "trpo":
            env = env.venv.venv.envs[0]
            runner(env,
                   policy_fn,
                   args.load_model_path,
                   timesteps_per_batch=args.num_length,
                   number_trajs=args.num_trajectories,
                   stochastic_policy=args.stochastic_policy,
                   save=args.save_sample,
                   reward_giver=reward_giver,
                   detect=args.detect,
                   dataset=dataset
                   )
        elif args.algo == "ppo":
            runner_ppo(env=env, load_model_path=args.load_model_path,
                       timesteps_per_batch=args.num_length,
                       number_trajs=args.num_trajectories,
                       stochastic_policy=args.stochastic_policy)
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, task_name=None, load_model_path=None, batch_size=1024, search=False, search_mode='traj', scene='all',
          p_update=True, rew_type=False, train_mode='all', gamma=0.995, r_norm=False, pretrain_std=False, still_std=False, max_iters=1000, model_output='DM'
          ):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                                 max_iters=BC_max_iter, pretrain_std=pretrain_std)

    if algo == 'trpo':
        from baselines.gail import trpo_runner
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        # env.seed(workerseed)
        trpo_runner.learn(env, policy_fn, reward_giver, dataset, rank,
                pretrained=pretrained, pretrained_weight=pretrained_weight,still_std=still_std,
                g_step=g_step, d_step=d_step,
                entcoeff=policy_entcoeff,
                max_timesteps=num_timesteps,
                max_iters=max_iters,
                ckpt_dir=checkpoint_dir, log_dir=log_dir,
                save_per_iter=save_per_iter,
                timesteps_per_batch=batch_size,
                max_kl=0.01, cg_iters=10, cg_damping=0.1,
                gamma=gamma, lam=0.97,
                vf_iters=5, vf_stepsize=1e-3,
                task_name=task_name, load_model_path=load_model_path, search=search, search_mode=search_mode,
                scene=scene,p_update=p_update,rew_type=rew_type,train_mode=train_mode,r_norm=r_norm,model_output=model_output)
    
    elif algo == "ppo":
        from baselines.ppo2.ppo2 import learn
        from baselines.ppo2.microbatched_model import MicrobatchedModel
        from functools import partial
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        eval_env = None
        if load_model_path != None:
            eval_env = copy(env)
        # env.seed(workerseed)
        # mujoco hyperparameters
        learn_fn = partial(learn, network='mlp', nsteps=batch_size, nminibatches=32, lam=0.95, gamma=0.99, \
                           noptepochs=10, log_interval=1, ent_coef=0.0, lr=lambda f: 3e-4 * f, cliprange=0.2, \
                           value_network='copy', total_timesteps=num_timesteps, seed=workerseed, still_std=still_std, \
                           save_interval=2, max_iters=max_iters, load_path=load_model_path, eval_env=eval_env)
        learn_fn(env=env)

    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False, reward_giver=None, detect=False, dataset=None):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    import pickle
    import time
    timenow = time.strftime("%Y%m%d%H%M%S", time.localtime())
    filename = '../log/'+timenow + '_easy.pkl'
    outfile = open(filename, 'wb')

    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_state(load_model_path)
    
    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
        ctrls = traj['ctrl']
    
        if detect:
            ob_batch = obs
            ac_batch = acs
            ob_expert, ac_expert = dataset.get_next_batch(len(ob_batch))
            error_g, g_total_number = reward_giver.detect_error(ob_batch, ac_batch, "g")
            error_e, e_total_number = reward_giver.detect_error(ob_expert, ac_expert, "e")
            print('Error generator: %f, total %f, error rate %f' % (error_g, g_total_number, float(error_g/g_total_number)))
            print('Error expert: %f, total %f, error rate %f' % (error_e, e_total_number, float(error_e/e_total_number)))
        if save:
            for i in range(len(traj['ob'])):
                ob = obs[i].tolist()
                ctrl = ctrls[i]
                ob_ctrl = ob+ctrl
                pickle.dump(ob_ctrl, outfile)        

    if stochastic_policy:   
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    #if save:
    #    filename = load_model_path.split('/')[-1] + '.' + env.spec.id
    #    np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
    #             lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret

def runner_ppo(env, load_model_path, timesteps_per_batch, number_trajs, stochastic_policy, num_timesteps=2e7, seed=0):

    from baselines.ppo2.ppo2 import evaluate
    from baselines.ppo2.microbatched_model import MicrobatchedModel
    from functools import partial

    # Set up for MPI seed
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    # mujoco hyperparameters
    evaluate_fn = partial(evaluate, network='mlp', nsteps=timesteps_per_batch, nminibatches=32, lam=0.95, gamma=0.99,
            noptepochs=10, ent_coef=0.0, lr=lambda f: 3e-4 * f, cliprange=0.2, value_network='copy',
            number_trajs=number_trajs, load_path=load_model_path,seed=workerseed)
    
    ep_len, ep_ret = evaluate_fn(env=env)
    print("Average length:", ep_len)
    print("Average return:", ep_ret)
    
    return ep_len, ep_ret

# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []
    ctrls = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)
        ob, rew, new, infos = env.step(ac)
        # env.render()
        rews.append(rew)
        ctrl = infos['control']
        ctrls.append(ctrl)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len, 'ctrl': ctrls}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
