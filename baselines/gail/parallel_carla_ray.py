'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import tensorflow as tf

import ray

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
import pickle

baseline_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, baseline_dir)

# # version 0.9.4
# # add carla config here, put carla directory as the parent directory of IDM
# carla_simulator_path = '/home/SENSETIME/maqiurui/reinforce/carla/carla_0.9.4/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg'
# try:
#     sys.path.append(carla_simulator_path)
#     sys.path.append(baseline_dir+'/../CARLA_0.9.4/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg')
#     sys.path.append(baseline_dir+'/../CARLA/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg')
# except IndexError:
#     pass

# version 0.9.5
# add carla config here, put carla directory as the parent directory of IDM
carla_simulator_path = '/home/SENSETIME/maqiurui/reinforce/carla/carla_0.9.5/PythonAPI/carla-0.9.5-py3.5-linux-x86_64.egg'
try:
    sys.path.append(carla_simulator_path)
    sys.path.append(baseline_dir+'/../CARLA_0.9.5/PythonAPI/carla-0.9.5-py3.5-linux-x86_64.egg')
    sys.path.append(baseline_dir+'/../CARLA/PythonAPI/carla-0.9.5-py3.5-linux-x86_64.egg')
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
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_monitor import VecMonitor
from copy import copy
from baselines.common import explained_variance, zipsame, dataset, fmt_row
from contextlib import contextmanager
from baselines.common import colorize
import time
from baselines.common.cg import cg
from collections import deque
from tensorflow.contrib.tensorboard.plugins import projector

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--sync', help='whether to sync server to client',action='store_true')
    parser.add_argument('--env_id', help='environment ID', default='Carla-Motion')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='../log/easy.pkl')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample', 'bc_evaluate', 'generate_data'], default='train')
    # for evaluatation
    #boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    #boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--stochastic_policy',action='store_true', help='use stochastic/deterministic policy to evaluate')
    parser.add_argument('--save_sample',action='store_true',help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=10, help='useless in carla')
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
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
    parser.add_argument('--num_trajectories', metavar='T', default=100, type=int, help='num of trajectories')
    parser.add_argument('--episode_length', metavar='EL', default=200, type=int, help='max length of a trajectory in rollout data')
    parser.add_argument('--num_length', metavar='L', default=200, type=int, help='max length of a trajectory in expert data')
    parser.add_argument('--batch_size', default=2048, type=int, help='timesteps per batch')
    parser.add_argument('--search', action='store_true', help='search for nearest expert path for training')
    parser.add_argument('--stack', default=1, type=int, help='stack frames')
    parser.add_argument('--search_mode', default='traj', type=str, help='search mode, default is nearest trajectory')
    parser.add_argument('--scene', default='all', type=str, choices=['all', 'straight', 'curve'], help='training scene')
    parser.add_argument('--mode', default='wp_obj', type=str, choices=['all', 'wp', 'wp_obj'], help='visible ranges for observation')
    parser.add_argument('--speed_mode', default='mix', type=str,
                        help='speed mode')
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
    parser.add_argument('--sigma_pos', default=0.4,type=float,help='sigma for track position reward, model as gaussian distribution')
    parser.add_argument('--sigma_vel_upper', default=3.6,type=float,help='sigma for velocity reward')
    parser.add_argument('--sigma_vel_lower', default=3.6,type=float,help='sigma for velocity reward')
    parser.add_argument('--sigma_ang', default=0.4,type=float,help='sigma for angle reward')
    parser.add_argument('--curriculumn_threshold', default=5,type=float,help='sigma for angle reward')
    parser.add_argument('--other_cars',default=6,type=int,help='the number of other cars')
    parser.add_argument('--model_output', default='DM', type=str, choices=['DM', 'CL'], help='model output, DM for the decision output, CL for the control output')
    parser.add_argument('--excute_mode', default='short', type=str, choices=['short', 'long'], help='excute mode for the decision model')
    parser.add_argument('--lanes', default=3, type=int, choices=[1, 3, 5, 7], help='total lane numbers')
    parser.add_argument('--overtake_curriculum', default=0, type=int, help='overtake curriculum botton')
    parser.add_argument('--flag', default='default', type=str, help='excute mode for the decision model')
    parser.add_argument('--scenario', action='store_true', help='Whether to use the scenario runner')
    parser.add_argument('--scenario_name', default='OtherLeadingVehicle', type=str, help='Scenarios')
    parser.add_argument('--p_pos', default=0., type=float, choices=[0., 1.0, 0.1, 0.01, 0.001], help='Propotion parameter for the cte error in the PID controller')
    parser.add_argument('--p_vel', default=1.0, type=float, choices=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05], help='Propotion parameter for the longitudinal control in the PID controller')
    parser.add_argument('--actor_nums', default=1, type=int, choices=[1, 2, 4, 8, 16, 32, 64], help='Actor numbers')
    parser.add_argument('--replay', action='store_true',
                           help='Replay the supervised data')
    parser.add_argument('--checkkeys', default='ST_TL_KL40_TR_KMidlane30_KMidlane25', type=str, help='Initialization on keypoints')
    parser.add_argument('--g_scratch_eps', default=0, type=int, help='The episode to start the curriculum training on keypoints')
    parser.add_argument('--rule_based', action='store_true', help='Rule based decision module')
    parser.add_argument('--update_D', action='store_false', help='Rule based decision module')

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
    task_name += args.checkkeys
    task_name += '_'
    task_name += 'CTE%sVel%s' % (str(args.p_pos), str(args.p_vel))
    task_name += '%snenv' % len(args.host.split('_'))
    task_name += '_%sSKIP' % str(args.D_skip)
    task_name += '_%s' % args.scenario_name
    task_name += '_CurSp%s' % args.g_scratch_eps
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
    #PLayer = args.policy_hidden_layer * 2 + 1
    #task_name += 'D%s' % args.d_model
    task_name += 'Reg%s_' % args.region
    task_name += 'DLR%s_' % args.d_lrate
    # task_name += 'MaxD%s_' % args.log_dis_max
    # if args.search:
    #     task_name += 'Lim%s_' % args.traj_limitation
    # task_name += 'GSize%s_' % args.policy_hidden_size
    # task_name += 'PLay%s_' % PLayer
    # task_name += 'DSize%s_' % args.adversary_hidden_size
    #if args.d_model == "separate":
    #    adversary_hidden_layer = args.adversary_hidden_layer - 1
    #    DLayer = adversary_hidden_layer + 3
    #else:
    #    adversary_hidden_layer = args.adversary_hidden_layer
    #    DLayer = adversary_hidden_layer + 2
    #if args.resampling != 0:
    #    task_name += 'ReS_%s' % args.resampling
    task_name += 'Batch%s_' % args.batch_size
    task_name += '%s_' %args.scene
    task_name += 'G%s'%args.gamma
    task_name = task_name + ".G_" + str(args.g_step) + ".D_" + str(args.d_step) + \
        ".G_entcoeff_" + str(args.policy_entcoeff) + ".D_entcoeff_" + str(args.adversary_entcoeff) + \
        ".maxkl_" + str(args.max_kl)
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

@ray.remote(num_gpus=1)
class Actor(object):
    def __init__(self, args, actor_id):
        # ------------------------------------------------------------------------
        # initializing env and arguments
        U.make_session(num_cpu=1).__enter__()
        set_global_seeds(args.seed+1000*actor_id)
        args = recover_args(args)
        self.actor_id = actor_id

        self.host = host = args.hosts[actor_id]
        port = args.ports[actor_id]
        assert len(host)==len(port), 'number of hosts and ports should match'
        logger.p_pos = args.p_pos
        logger.p_vel = args.p_vel
        logger.keypoints = {"ST": [0], "TL": [10], "KL40": [30], "TR": [40], "KMidlane30": [66], "KMidlane25": [100]}
        logger.keyframes = [0, 10, 30, 40, 66, 100]
        logger.task = args.task
        logger.scenario = args.scenario
        logger.scenario_name = args.scenario_name
        logger.actor_id = actor_id
        checkkeys = args.checkkeys.split("_")
        logger.checkkeys = checkkeys

        def make_env(host, port, args):
            client = carla.Client(host, port)
            client.set_timeout(600.0)
            carla_world = client.get_world()
            assert not(args.A_skip != 1 and args.D_skip != 1)
            world = World(carla_world, args.sync, args.sigmas, camera=args.render, A_skip=args.A_skip, mode=args.mode, feature=args.feature,
                          dim=args.dim, dis_max=args.dis_max, spawn_mode=args.spawn_mode, render=args.render, width=args.width, height=args.height,
                          other_cars=args.other_cars, curriculumn_threshold=args.curriculumn_threshold, max_lanes=args.lanes,
                          scenario_name=args.scenario_name, client=client, host=host, port=port, scenario=args.scenario, checkkeys=checkkeys)
            args.log_dis_max = int(args.dis_max**16)

            if args.task == "evaluate":
                test = True
            else:
                test = False

            if args.model_output == 'DM':
                env = CarlaDM(world, args.episode_length, args.stack, args.train_mode, test, args.region, start_v=args.start_v, excute_mode=args.excute_mode,
                              D_skip=args.D_skip, overtake_curriculum=args.overtake_curriculum, scenario_name=args.scenario_name, g_scratch_eps=args.g_scratch_eps,
                              rule_based=args.rule_based)
            elif args.model_output == 'CL':
                env = Carla(world, args.episode_length, args.stack, args.train_mode, test, args.region, start_v=args.start_v, scenario_name=args.scenario_name)
            gym.logger.setLevel(logging.WARN)

            return env

        # @Junning: menv wrapper
        envs = [make_env(host[i], port[i], args) for i in range(len(host))]
        env = DummyCarlaEnv(envs)

        # wrapper for stacking frames
        env = VecFrameStack(env, args.stack)
        env = VecMonitor(env)

        class ENV(object):
            def __init__(self, env):
                self.observation_space = env.observation_space
                self.action_space = env.action_space
        args.env = ENV(env)

        def policy_fn(name, ob_space, ac_space, reuse=False):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                        reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=args.policy_hidden_layer)
        args.policy_func = policy_fn
        task_name = get_task_name(args)
        args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
        args.log_dir = osp.join(args.log_dir, task_name)
        logger.init_std = args.init_std

        datashape = env.observation_space.shape[0]
        dataset = None

        if not args.rew_type or args.pretrained: # Using TRPO
            if args.env_id == "Carla-Motion":
                dataset = Carla_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation, num_trajectories=args.num_trajectories, num_length=args.num_length, data_shape=datashape, search=args.search, mode=args.mode, feature=args.feature, train_mode=args.train_mode, dim=args.dim, resampling=args.resampling, episode_length=args.episode_length)
            else:
                dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        logger.num_length = args.num_length
        if dataset is not None:
            logger.expert_acs = copy(dataset.acs_all[:args.num_length])      

        dis_name = "dis_%d" % actor_id
        self.reward_giver = reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff, adversary_hidden_layer=args.adversary_hidden_layer, sn=args.sn, d_actv=args.d_actv, lr_rate=args.d_lrate, model=args.d_model, train_mode=args.train_mode, dim=args.dim, scope=dis_name)

        # ----------------------------------------------------------------------------
        # initializing policy
        ob_space = env.observation_space
        ac_space = env.action_space
        policy_func = policy_fn
        pretrained_weight = None
        pi_name = "pi_%d" % actor_id
        self.pi_name = pi_name
        self.pi = pi = policy_func(pi_name, ob_space, ac_space, reuse=(pretrained_weight != None))
        # ----------------------------------------------------------------------------
        # initializing model
        from baselines.gail.trpo_runner import Model, Runner
        if args.load_model_path is not None and args.task == 'evaluate' or args.task == 'generate_data':
            stochastic = args.stochastic_policy
        else:
            stochastic = True
        print("Stochastic Policy: ", stochastic)
        model = Model(pi, reward_giver, env.num_envs, stochastic=stochastic)
        # ----------------------------------------------------------------------------
        # initializing runner
        expert_dataset = dataset
        args.expert_dataset = expert_dataset
        if expert_dataset is not None:
            self.seg_gen = seg_gen = Runner(env, model, args.batch_size, gamma=args.gamma, lam=0.97, length=expert_dataset.num_length, rew_type=args.rew_type, model_output=args.model_output)
        else:
            self.seg_gen = seg_gen = Runner(env, model, args.batch_size, gamma=args.gamma, lam=0.97, length=logger.num_length, rew_type=args.rew_type, model_output=args.model_output)
        # ----------------------------------------------------------------------------
        # initializing parameters-updated operator
        self.params_pi = self.pi.get_variables()
        self.params_dis = self.reward_giver.get_trainable_variables()
        self.params_pi_placeholders = [tf.placeholder(shape=param.shape, dtype=param.dtype) for param in self.params_pi]
        self.params_dis_placeholders = [tf.placeholder(shape=param.shape, dtype=param.dtype) for param in self.params_dis]
        self.assign_params_pi = [tf.assign(param, param_new) for param, param_new in zip(pi.get_variables(), self.params_pi_placeholders)]
        self.assign_params_dis = [tf.assign(param, param_new) for param, param_new in zip(reward_giver.get_trainable_variables(), self.params_dis_placeholders)]

        args.wp_len = logger.wp_len
        args.obj_len = logger.obj_len
        args.road_len = logger.road_len
        args.ctrl_len = logger.ctrl_len

        args.egoa_idx = logger.egoa_idx
        args.egov_idx = logger.egov_idx
        args.zombiea_idx = logger.zombiea_idx
        args.zombiev_idx = logger.zombiev_idx
        args.zombiebx_idx = logger.zombiebx_idx

        self.args = args
        U.initialize()


    def get_params(self):
        return [self.pi.get_variables(), self.reward_giver.get_trainable_variables()]

    def update_params(self, params):
        # note that params should be numpy array
        params_pi, params_dis = params[0], params[1]
        try:
            sess = tf.get_default_session()
            sess.run(self.assign_params_pi, feed_dict={pholder: param_pi for pholder, param_pi in zip(self.params_pi_placeholders, params_pi)})
            sess.run(self.assign_params_dis, feed_dict={pholder: param_dis for pholder, param_dis in zip(self.params_dis_placeholders, params_dis)})
            return "Succeed"
        except:
            return "Failed"

    def get_args(self):
        return self.args


    def get_batch(self):
        @contextmanager
        def timed(msg):
            rank = MPI.COMM_WORLD.Get_rank()
            if rank == 0:
                print(colorize(msg, color='magenta'))
                tstart = time.time()
                yield
                print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
            else:
                yield

        with timed("sampling"):
            batch = self.seg_gen.run()
        log_print = "================= %s =================" %(str(self.host))
        logger.log(log_print)
        return batch

from gym import spaces
from baselines.common.mpi_adam import MpiAdam
@ray.remote(num_gpus=1)
class Learner(object):
    def __init__(self, args):
        self.g_steps_now = 0
        self.args = args
        set_global_seeds(args.seed)
        logger.keypoints = {"ST": [0], "TL": [10], "KL40": [30], "TR": [40], "KMidlane30": [66], "KMidlane25": [100]}
        logger.keyframes = [0, 10, 30, 40, 66, 100]
        logger.ckpt_dir_name = args.checkpoint_dir
        logger.configure(args.checkpoint_dir, format_strs=['stdout', 'json', 'log', 'tensorboard'])
        logger.log('Spawn mode: '+str(args.spawn_mode))
        logger.log('Trajs: '+str(args.num_trajectories))
        logger.log('Rollout data epsidoe Length: '+str(args.episode_length))
        logger.log('Expert data epsidoe Length: '+str(args.num_length))
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

        logger.wp_len = args.wp_len
        logger.obj_len = args.obj_len
        logger.road_len = args.road_len
        logger.ctrl_len = args.ctrl_len

        logger.egoa_idx = args.egoa_idx
        logger.egov_idx = args.egov_idx
        logger.zombiea_idx = args.zombiea_idx
        logger.zombiev_idx = args.zombiev_idx
        logger.zombiebx_idx = args.zombiebx_idx

        logger.init_std = args.init_std
        adversary_hidden_layer = args.adversary_hidden_layer
        DLayer = adversary_hidden_layer + 2
        logger.log('DLay: '+str(DLayer))
        logger.log('DAtv: '+str(args.d_actv))
        self.expert_dataset = args.expert_dataset
        self.nworkers = MPI.COMM_WORLD.Get_size()
        self.rank = rank = MPI.COMM_WORLD.Get_rank()
        np.set_printoptions(precision=3)

        # Setup losses and stuff
        # ----------------------------------------
        env = args.env
        policy_func = args.policy_func
        self.reward_giver = reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff, adversary_hidden_layer=args.adversary_hidden_layer, sn=args.sn, d_actv=args.d_actv, lr_rate=args.d_lrate, model=args.d_model, train_mode=args.train_mode, dim=args.dim, scope="Learner")
        ob_space = env.observation_space
        ac_space = env.action_space
        pretrained_weight = None
        BC_max_iter = 10000
        if self.args.pretrained and (BC_max_iter > 0):
            # Pretrain with behavior cloning
            from baselines.gail import behavior_clone
            pretrained_weight = behavior_clone.learn(env, policy_func, self.expert_dataset,
                                                     max_iters=BC_max_iter, pretrain_std=self.args.pretrain_std)


        self.pi = pi = policy_func("pi_learner", ob_space, ac_space, reuse=(pretrained_weight != None))
        self.oldpi = oldpi = policy_func("oldpi_learner", ob_space, ac_space)
        atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

        ob = U.get_placeholder_cached(name="ob")
        ac = pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)

        entbonus = args.policy_entcoeff * meanent

        vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
        surrgain = tf.reduce_mean(ratio * atarg)

        optimgain = surrgain + entbonus

        if isinstance(env.action_space, spaces.MultiDiscrete):
            losses = [optimgain, meankl, entbonus, surrgain, meanent]
            self.loss_names = loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]
        elif isinstance(env.action_space, spaces.Box):
            pi_mean = tf.reduce_mean(pi.pd.mean)
            pi_std = tf.reduce_mean(pi.pd.std)

            steer = tf.reduce_mean(pi.pd.mean[:, 0])
            steer_std = tf.reduce_mean(pi.pd.std[:, 0])
            if args.train_mode == "all":
                throttle_brake = tf.reduce_mean(pi.pd.mean[:, 1])
                throttle_brake_std = tf.reduce_mean(pi.pd.std[:, 1])
                losses = [optimgain, meankl, entbonus, surrgain, meanent, pi_mean, pi_std, steer, throttle_brake, steer_std, throttle_brake_std]
                self.loss_names = loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy", "pi_mean", "pi_std", "steer", "throttle_brake", "steer_std", "throttle_brake_std"]
            elif args.train_mode == "steer":
                losses = [optimgain, meankl, entbonus, surrgain, meanent, pi_mean, pi_std, steer]
                self.loss_names = loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy", "pi_mean", "pi_std", "steer"]

        dist = meankl

        all_var_list = pi.get_trainable_variables()
        if args.still_std:
            var_list = [v for v in all_var_list if v.name.startswith("pi_learner/pol")]# or v.name.startswith("pi/logstd")]
        else:
            var_list = [v for v in all_var_list if v.name.startswith("pi_learner/pol") or v.name.startswith("pi_learner/logstd")]

        vf_var_list = [v for v in all_var_list if v.name.startswith("pi_learner/vff")]
        self.d_adam = d_adam = MpiAdam(reward_giver.get_trainable_variables())
        self.vfadam = vfadam = MpiAdam(vf_var_list)

        self.get_flat = get_flat = U.GetFlat(var_list)
        self.set_from_flat = set_from_flat = U.SetFromFlat(var_list)
        klgrads = tf.gradients(dist, var_list)
        flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        shapes = [var.get_shape().as_list() for var in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            sz = U.intprod(shape)
            tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
            start += sz
        gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
        fvp = U.flatgrad(gvp, var_list)

        self.assign_old_eq_new = assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        self.compute_losses = compute_losses = U.function([ob, ac, atarg], losses)
        self.compute_lossandgrad = compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
        self.compute_fvp = compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
        self.compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

        U.initialize()

        # ----------------------------------------------------------------------------
        # load model
        if args.load_model_path is not None:
            self.load_model()
        # ----------------------------------------------------------------------------

        th_init = get_flat()
        MPI.COMM_WORLD.Bcast(th_init, root=0)
        set_from_flat(th_init)
        d_adam.sync()
        vfadam.sync()
        if rank == 0:
            print("Init param sum", th_init.sum(), flush=True)

        self.saver = tf.train.Saver(max_to_keep=5000)
        self.saver_best = tf.train.Saver()
        model_init = os.path.join(self.args.checkpoint_dir+'/models/', 'model_init')
        self.saver.save(tf.get_default_session(), model_init)
        self.eptruerew_best = 0

        self.episodes_so_far = 0
        self.timesteps_so_far = 0
        self.iters_so_far = 0
        self.tstart = time.time()
        self.lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards
        self.true_rewbuffer = deque(maxlen=100)

        self.ep_vs_global = deque(maxlen=40)
        self.ep_rets_global = deque(maxlen=40)
        self.ep_true_rets_global = deque(maxlen=40)

    def learn(self, batches, max_kl=0.01, cg_iters=10, cg_damping=0.1, vf_iters=5, vf_stepsize=1e-3, d_stepsize=3e-4):
        @contextmanager
        def timed(msg):
            if self.rank == 0:
                print(colorize(msg, color='magenta'))
                tstart = time.time()
                yield
                print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
            else:
                yield

        def allmean(x):
            assert isinstance(x, np.ndarray)
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= self.nworkers
            return out

        def fisher_vector_product(p):
            return allmean(self.compute_fvp(p, *fvpargs)) + cg_damping * p

        def outstr(v_x, v_y, a_x, a_y, steer, throttle):
            return "Vx"+str(round(v_x, 1))+"_"+"Vy"+str(round(v_y, 1))+"_"+"Ax"+str(round(a_x, 1))+"_"+"Ay"+str(round(a_y, 1))+"_"+"St"+str(round(steer, 1))+"_"+"Th"+str(round(throttle, 1))

        #for batch in batches:
        seg = batches
        logger.log("Optimizing Policy...")
 
        for _ in range(1):
            # reward normalization
            # if r_norm:
            #     rew_norm.update(seg["rew"])
            #     seg["rew"] = (seg["rew"] - rew_norm.mean) / rew_norm.var

            ob, current_pos, yaw, ac, atarg, tdlamret, done = seg["ob"], seg["current_pos"], seg["yaw"], seg["ac"], seg[
                "adv"], seg["tdlamret"], seg["new"]
            ob_start = ob[0]
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(self.pi, "ob_rms"): self.pi.ob_rms.update(ob)  # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            self.assign_old_eq_new()  # set old parameter values to new parameter values
            
            if self.args.p_update:
                with timed("computegrad"):
                    *lossbefore, g = self.compute_lossandgrad(*args)
                lossbefore = allmean(np.array(lossbefore))
                g = allmean(g)
                g_policy = g
                if np.allclose(g, 0):
                    logger.log("Got zero gradient. not updating")
                else:
                    with timed("cg"):
                        stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=self.rank == 0)
                    assert np.isfinite(stepdir).all()
                    shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                    lm = np.sqrt(shs / max_kl)
                    # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                    fullstep = stepdir / lm
                    expectedimprove = g.dot(fullstep)
                    surrbefore = lossbefore[0]

                    stepsize = 1.0
                    thbefore = self.get_flat()
                    for _ in range(10):
                        thnew = thbefore + fullstep * stepsize
                        self.set_from_flat(thnew)
                        meanlosses = surr, kl, *_ = allmean(np.array(self.compute_losses(*args)))
                        improve = surr - surrbefore
                        logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                        if not np.isfinite(meanlosses).all():
                            logger.log("Got non-finite value of losses -- bad!")
                        elif kl > max_kl * 1.5:
                            logger.log("violated KL constraint. shrinking step.")
                        elif improve < 0:
                            logger.log("surrogate didn't improve. shrinking step.")
                        else:
                            logger.log("Stepsize OK!")
                            break
                        stepsize *= .5
                    else:
                        logger.log("couldn't compute a good step")
                        self.set_from_flat(thbefore)
                    if self.nworkers > 1 and iters_so_far % 20 == 0:
                        paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum()))  # list of tuples
                        assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
                with timed("vf"):
                    for _ in range(vf_iters):
                        for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                                include_final_partial_batch=False, batch_size=128):
                            if hasattr(self.pi, "ob_rms"):
                                self.pi.ob_rms.update(mbob)  # update running mean/std for policy
                            g = allmean(self.compute_vflossandgrad(mbob, mbret))
                            self.vfadam.update(g, vf_stepsize)
                    g_vf = g
                g_losses = meanlosses
                for (lossname, lossval) in zip(self.loss_names, meanlosses):
                    logger.record_tabular(lossname, lossval)
                logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
                ep_v, ep_acc, ep_left_offset, ep_right_offset = seg["ep_v"], seg["ep_acc"], seg["ep_left_offset"], seg["ep_right_offset"]
        self.g_steps_now = self.g_steps_now + 1

        if self.expert_dataset is not None and self.g_steps_now == self.args.g_step:
            logger.log("Optimizing Discriminator...")
            logger.log(fmt_row(13, self.reward_giver.loss_name))
            ob_expert, ac_expert = self.expert_dataset.get_next_batch(len(ob))
            batch_size = len(ob) // self.args.d_step
            d_losses = []  # list of tuples, each of which gives the loss for a minibatch
            if self.args.model_output == 'DM':
                ac = seg["ctrl"]
            for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                        include_final_partial_batch=False,
                                                        batch_size=batch_size):
                if not self.args.p_update:
                    with timed("just update discriminator"):
                        ob_expert, ac_expert, search_prop = self.expert_dataset.obs, self.expert_dataset.acs, 0
                elif self.args.search:
                    with timed("searching batch"):
                        if search_mode == 'step':
                            ob_expert, ac_expert, search_prop = self.expert_dataset.search_batch_step(ob_batch, ac_batch)
                        elif search_mode == 'traj':
                            ob_expert, ac_expert, search_prop = self.expert_dataset.search_batch_traj(ob_start, batch_size, scene=self.args.scene)

                else:
                    ob_expert, ac_expert = self.expert_dataset.get_next_batch(len(ob_batch), scene=self.args.scene)
                # update running mean/std for reward_giver
                if self.args.update_D:
                    if hasattr(self.reward_giver, "obs_rms"): self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
                    *newlosses, g = self.reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
                    g_d = g
                    self.d_adam.update(allmean(g), d_stepsize)
                    d_losses.append(newlosses)

            if self.args.update_D:
                logger.log(fmt_row(13, np.mean(d_losses, axis=0)))
                d_losses_name = self.reward_giver.loss_name
                d_losses_data = np.mean(d_losses, axis=0)
                kvs = [{name: data} for name, data in zip(d_losses_name, d_losses_data)]
                for kv in kvs:
                    for k, v in kv.items():
                        logger.record_tabular(k, v)
            self.g_steps_now = 0
            
        if self.g_steps_now == 0 or self.args.g_step==1:
            lrlocal = (seg["ep_true_lens"], seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            true_lens, lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
            self.true_rewbuffer.extend(true_rets)
            self.lenbuffer.extend(lens)
            self.rewbuffer.extend(rews)

            # Save model
            eptruerew_now = np.mean(true_rets)
            ckpt_dir = self.args.checkpoint_dir
            if self.rank == 0 and self.iters_so_far % self.args.save_per_iter == 0 and ckpt_dir is not None:
                modelname = 'model%d.ckpt'%self.iters_so_far
                fname = os.path.join(ckpt_dir+'/models/', modelname)
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                self.saver.save(tf.get_default_session(), fname)
            if self.rank == 0 and ckpt_dir is not None and eptruerew_now > self.eptruerew_best:
                modelname = 'modelbest.ckpt'
                fname = os.path.join(ckpt_dir+'/models', modelname)
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                self.saver_best.save(tf.get_default_session(), fname)
                self.eptruerew_best = eptruerew_now
            eptruerew_last = eptruerew_now
            logger.record_tabular("EpLenMean", np.mean(self.lenbuffer))
            logger.record_tabular("EpLenMax", np.max(self.lenbuffer))
            if self.expert_dataset is not None:
                logger.record_tabular("EpRewMean", np.mean(self.rewbuffer))
                logger.record_tabular("EpRewMax", np.max(self.rewbuffer))
            logger.record_tabular("EpTrueRewMean", np.mean(self.true_rewbuffer))
            logger.record_tabular("EpTrueRewMax", np.max(self.true_rewbuffer))
            logger.record_tabular("EpThisIter", len(true_lens))
            if self.args.p_update:
                logger.record_tabular("EpVelocity", ep_v)
                logger.record_tabular("EpAcc", ep_acc)
                logger.record_tabular("EpLeftOffset", ep_left_offset)
                logger.record_tabular("EpRightOffset", ep_right_offset)

            corr_rew = np.corrcoef([seg["rew"], seg["truerew"]])[0][1]

            ep_rets = [ret for ret in seg["ep_rets"]]
            min_len = min(len(seg["v_ep"]), len(seg["ep_true_rets"]), len(ep_rets))
            for i in range(min_len):
                self.ep_vs_global.append(seg["v_ep"][i])
                self.ep_rets_global.append(ep_rets[i])
                self.ep_true_rets_global.append(seg["ep_true_rets"][i])
            corr_eprew = np.corrcoef([self.ep_vs_global, self.ep_rets_global])[0][1]
            corr_eptruerew = np.corrcoef([self.ep_vs_global, self.ep_true_rets_global])[0][1]
            logger.record_tabular("CorrRew", corr_rew)
            logger.record_tabular("CorrEpRew", corr_eprew)
            logger.record_tabular("CorrEpTrueRew", corr_eptruerew)

            self.episodes_so_far += len(true_lens)
            self.timesteps_so_far += sum(true_lens)
            self.iters_so_far += 1

            logger.record_tabular("EpisodesSoFar", self.episodes_so_far)
            logger.record_tabular("TimestepsSoFar", self.timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - self.tstart)
            logger.log("")
            
            # --------------------------------------------------------
            # evaluate module
            # --------------------------------------------------------
            if not self.args.rew_type and self.args.scenario:
                keypoints = logger.keypoints

                v_x_expert_dict = dict();
                v_y_expert_dict = dict();
                a_x_expert_dict = dict();
                a_y_expert_dict = dict()
                v_x_generator_dict = dict();
                v_y_generator_dict = dict();
                a_x_generator_dict = dict();
                a_y_generator_dict = dict()
                v_x_relative_dict = dict();
                v_y_relative_dict = dict();
                a_x_relative_dict = dict();
                a_y_relative_dict = dict()

                v_x_expert_zombie_0_dict = dict();
                v_y_expert_zombie_0_dict = dict();
                a_x_expert_zombie_0_dict = dict();
                a_y_expert_zombie_0_dict = dict()
                v_x_generator_zombie_0_dict = dict();
                v_y_generator_zombie_0_dict = dict();
                a_x_generator_zombie_0_dict = dict();
                a_y_generator_zombie_0_dict = dict()
                v_x_relative_zombie_0_dict = dict();
                v_y_relative_zombie_0_dict = dict();
                a_x_relative_zombie_0_dict = dict();
                a_y_relative_zombie_0_dict = dict()

                v_x_expert_zombie_1_dict = dict();
                v_y_expert_zombie_1_dict = dict();
                a_x_expert_zombie_1_dict = dict();
                a_y_expert_zombie_1_dict = dict()
                v_x_generator_zombie_1_dict = dict();
                v_y_generator_zombie_1_dict = dict();
                a_x_generator_zombie_1_dict = dict();
                a_y_generator_zombie_1_dict = dict()
                v_x_relative_zombie_1_dict = dict();
                v_y_relative_zombie_1_dict = dict();
                a_x_relative_zombie_1_dict = dict();
                a_y_relative_zombie_1_dict = dict()

                box_p0_x_expert_zombie_0_dict = dict();
                box_p0_y_expert_zombie_0_dict = dict();
                box_p1_x_expert_zombie_0_dict = dict();
                box_p1_y_expert_zombie_0_dict = dict();
                box_p2_x_expert_zombie_0_dict = dict();
                box_p2_y_expert_zombie_0_dict = dict();
                box_p3_x_expert_zombie_0_dict = dict();
                box_p3_y_expert_zombie_0_dict = dict()
                box_p0_x_generator_zombie_0_dict = dict();
                box_p0_y_generator_zombie_0_dict = dict();
                box_p1_x_generator_zombie_0_dict = dict();
                box_p1_y_generator_zombie_0_dict = dict();
                box_p2_x_generator_zombie_0_dict = dict();
                box_p2_y_generator_zombie_0_dict = dict();
                box_p3_x_generator_zombie_0_dict = dict();
                box_p3_y_generator_zombie_0_dict = dict()
                box_p0_x_relative_zombie_0_dict = dict();
                box_p0_y_relative_zombie_0_dict = dict();
                box_p1_x_relative_zombie_0_dict = dict();
                box_p1_y_relative_zombie_0_dict = dict();
                box_p2_x_relative_zombie_0_dict = dict();
                box_p2_y_relative_zombie_0_dict = dict();
                box_p3_x_relative_zombie_0_dict = dict();
                box_p3_y_relative_zombie_0_dict = dict()

                box_p0_x_expert_zombie_1_dict = dict();
                box_p0_y_expert_zombie_1_dict = dict();
                box_p1_x_expert_zombie_1_dict = dict();
                box_p1_y_expert_zombie_1_dict = dict();
                box_p2_x_expert_zombie_1_dict = dict();
                box_p2_y_expert_zombie_1_dict = dict();
                box_p3_x_expert_zombie_1_dict = dict();
                box_p3_y_expert_zombie_1_dict = dict()
                box_p0_x_generator_zombie_1_dict = dict();
                box_p0_y_generator_zombie_1_dict = dict();
                box_p1_x_generator_zombie_1_dict = dict();
                box_p1_y_generator_zombie_1_dict = dict();
                box_p2_x_generator_zombie_1_dict = dict();
                box_p2_y_generator_zombie_1_dict = dict();
                box_p3_x_generator_zombie_1_dict = dict();
                box_p3_y_generator_zombie_1_dict = dict()
                box_p0_x_relative_zombie_1_dict = dict();
                box_p0_y_relative_zombie_1_dict = dict();
                box_p1_x_relative_zombie_1_dict = dict();
                box_p1_y_relative_zombie_1_dict = dict();
                box_p2_x_relative_zombie_1_dict = dict();
                box_p2_y_relative_zombie_1_dict = dict();
                box_p3_x_relative_zombie_1_dict = dict();
                box_p3_y_relative_zombie_1_dict = dict()

                abs_x_expert_dict = dict();
                abs_y_expert_dict = dict();
                abs_z_expert_dict = dict();
                yaw_expert_dict = dict()
                abs_x_generator_dict = dict();
                abs_y_generator_dict = dict();
                abs_z_generator_dict = dict();
                yaw_generator_dict = dict()
                abs_x_relative_dict = dict();
                abs_y_relative_dict = dict();
                abs_z_relative_dict = dict();
                yaw_relative_dict = dict()

                statistic_len = 40

                for key in keypoints.keys():
                    stage = key
                    for keypoint in keypoints[stage]:
                        # saliency test
                        ob_expert_evaluate, ac_expert_evaluate, abspos_yaw_expert_evaluate = self.expert_dataset.obs_all[
                                                                                                 keypoint].reshape(1, -1), \
                                                                                             self.expert_dataset.acs_all[
                                                                                                 keypoint].reshape(1, -1), \
                                                                                             self.expert_dataset.obs_abspos_yaw_all[
                                                                                                 keypoint].reshape(1, -1)

                        ob_generator_evaluate, ac_generator_evaluate = ob_batch[0].reshape(1, -1), ac_batch[0].reshape(1,
                                                                                                                       -1)
                        saliencies = self.reward_giver.get_saliency(ob_expert_evaluate, ac_expert_evaluate,
                                                                    ob_generator_evaluate, ac_generator_evaluate)
                        # ob_batch.shape (256, 306)
                        for k, v in saliencies.items():
                            #    logger.record_tabular(k, v)
                            logger.record_tabular(stage + "_" + str(keypoint) + "_" + k, v)

                        # counterfactual test
                        v_x, v_y = logger.egov_idx[0], logger.egov_idx[0]+1
                        a_x, a_y = logger.egoa_idx[0], logger.egoa_idx[0]+1

                        zombie_0_v_x, zombie_0_v_y, zombie_1_v_x, zombie_1_v_y = logger.zombiev_idx[0], logger.zombiev_idx[
                            0] + 1, logger.zombiev_idx[0] + 3, logger.zombiev_idx[0] + 4  # 3 x 6
                        zombie_0_a_x, zombie_0_a_y, zombie_1_a_x, zombie_1_a_y = logger.zombiea_idx[0], logger.zombiea_idx[
                            0] + 1, logger.zombiea_idx[0] + 3, logger.zombiea_idx[0] + 4  # 3 x 6

                        zombie_0_box_p0_x, zombie_0_box_p0_y, zombie_0_box_p0_z, zombie_0_box_p1_x, zombie_0_box_p1_y, zombie_0_box_p1_z, zombie_0_box_p2_x, zombie_0_box_p2_y, zombie_0_box_p2_z, zombie_0_box_p3_x, zombie_0_box_p3_y, zombie_0_box_p3_z = \
                            logger.zombiebx_idx[0], logger.zombiebx_idx[0] + 1, logger.zombiebx_idx[0] + 2, \
                            logger.zombiebx_idx[
                                0] + 3, logger.zombiebx_idx[0] + 4, logger.zombiebx_idx[0] + 5, logger.zombiebx_idx[0] + 6, \
                            logger.zombiebx_idx[0] + 7, logger.zombiebx_idx[0] + 8, logger.zombiebx_idx[0] + 9, \
                            logger.zombiebx_idx[0] + 10, logger.zombiebx_idx[0] + 11  # 12 x 6
                        zombie_1_box_p0_x, zombie_1_box_p0_y, zombie_1_box_p0_z, zombie_1_box_p1_x, zombie_1_box_p1_y, zombie_1_box_p1_z, zombie_1_box_p2_x, zombie_1_box_p2_y, zombie_1_box_p2_z, zombie_1_box_p3_x, zombie_1_box_p3_y, zombie_1_box_p3_z = \
                            logger.zombiebx_idx[0] + 12, logger.zombiebx_idx[0] + 13, logger.zombiebx_idx[0] + 14, \
                            logger.zombiebx_idx[0] + 15, logger.zombiebx_idx[0] + 16, logger.zombiebx_idx[0] + 17, \
                            logger.zombiebx_idx[0] + 18, logger.zombiebx_idx[0] + 19, logger.zombiebx_idx[0] + 20, \
                            logger.zombiebx_idx[0] + 21, logger.zombiebx_idx[0] + 22, logger.zombiebx_idx[0] + 23  # 12 x 6

                        # print('v_x: ', v_x, 'v_y: ', v_y, 'a_x: ', a_x, 'a_y: ', a_y, 'logger.zombiev_idx[0]: ', logger.zombiev_idx[0], logger.zombiev_idx[0]+4, 'logger.zombiea_idx[0]: ', logger.zombiea_idx[0], logger.zombiea_idx[0]+4,
                        #       'logger.zombiebx_idx[0]: ', logger.zombiebx_idx[0], logger.zombiebx_idx[0]+11, 'logger.zombiebx_idx[0]: ', logger.zombiebx_idx[0]+12, logger.zombiebx_idx[0]+23,)
                        # # abs_x_idx, abs_y_idx, abs_z_idx, yaw_idx = logger.zombiev_idx[1], logger.zombiev_idx[1]+1, logger.zombiev_idx[1]+2, logger.zombiev_idx[1]+3

                        all_dict = [v_x_expert_dict, v_y_expert_dict, a_x_expert_dict, a_y_expert_dict, v_x_generator_dict,
                                    v_y_generator_dict, a_x_generator_dict, a_y_generator_dict, v_x_relative_dict,
                                    v_y_relative_dict, a_x_relative_dict, a_y_relative_dict,
                                    v_x_expert_zombie_0_dict, v_y_expert_zombie_0_dict, a_x_expert_zombie_0_dict,
                                    a_y_expert_zombie_0_dict,
                                    v_x_generator_zombie_0_dict, v_y_generator_zombie_0_dict, a_x_generator_zombie_0_dict,
                                    a_y_generator_zombie_0_dict,
                                    v_x_relative_zombie_0_dict, v_y_relative_zombie_0_dict, a_x_relative_zombie_0_dict,
                                    a_y_relative_zombie_0_dict,
                                    v_x_expert_zombie_1_dict, v_y_expert_zombie_1_dict, a_x_expert_zombie_1_dict,
                                    a_y_expert_zombie_1_dict,
                                    v_x_generator_zombie_1_dict, v_y_generator_zombie_1_dict, a_x_generator_zombie_1_dict,
                                    a_y_generator_zombie_1_dict,
                                    v_x_relative_zombie_1_dict, v_y_relative_zombie_1_dict, a_x_relative_zombie_1_dict,
                                    a_y_relative_zombie_1_dict,
                                    box_p0_x_expert_zombie_0_dict, box_p0_y_expert_zombie_0_dict,
                                    box_p1_x_expert_zombie_0_dict, box_p1_y_expert_zombie_0_dict,
                                    box_p2_x_expert_zombie_0_dict, box_p2_y_expert_zombie_0_dict,
                                    box_p3_x_expert_zombie_0_dict, box_p3_y_expert_zombie_0_dict,
                                    box_p0_x_generator_zombie_0_dict, box_p0_y_generator_zombie_0_dict,
                                    box_p1_x_generator_zombie_0_dict, box_p1_y_generator_zombie_0_dict,
                                    box_p2_x_generator_zombie_0_dict, box_p2_y_generator_zombie_0_dict,
                                    box_p3_x_generator_zombie_0_dict, box_p3_y_generator_zombie_0_dict,
                                    box_p0_x_relative_zombie_0_dict, box_p0_y_relative_zombie_0_dict,
                                    box_p1_x_relative_zombie_0_dict, box_p1_y_relative_zombie_0_dict,
                                    box_p2_x_relative_zombie_0_dict, box_p2_y_relative_zombie_0_dict,
                                    box_p3_x_relative_zombie_0_dict, box_p3_y_relative_zombie_0_dict,
                                    box_p0_x_expert_zombie_1_dict, box_p0_y_expert_zombie_1_dict,
                                    box_p1_x_expert_zombie_1_dict, box_p1_y_expert_zombie_1_dict,
                                    box_p2_x_expert_zombie_1_dict,
                                    box_p2_y_expert_zombie_1_dict, box_p3_x_expert_zombie_1_dict,
                                    box_p3_y_expert_zombie_1_dict,
                                    box_p0_x_generator_zombie_1_dict, box_p0_y_generator_zombie_1_dict,
                                    box_p1_x_generator_zombie_1_dict, box_p1_y_generator_zombie_1_dict,
                                    box_p2_x_generator_zombie_1_dict, box_p2_y_generator_zombie_1_dict,
                                    box_p3_x_generator_zombie_1_dict, box_p3_y_generator_zombie_1_dict,
                                    box_p0_x_relative_zombie_1_dict, box_p0_y_relative_zombie_1_dict,
                                    box_p1_x_relative_zombie_1_dict, box_p1_y_relative_zombie_1_dict,
                                    box_p2_x_relative_zombie_1_dict, box_p2_y_relative_zombie_1_dict,
                                    box_p3_x_relative_zombie_1_dict, box_p3_y_relative_zombie_1_dict,
                                    abs_x_expert_dict, abs_y_expert_dict, abs_z_expert_dict, yaw_expert_dict,
                                    abs_x_generator_dict, abs_y_generator_dict, abs_z_generator_dict, yaw_generator_dict,
                                    abs_x_relative_dict, abs_y_relative_dict, abs_z_relative_dict, yaw_relative_dict,
                                    ]

                        for each_dict in all_dict:
                            if keypoint not in each_dict:
                                each_dict[keypoint] = list()

                        v_x_expert, v_y_expert, a_x_expert, a_y_expert = ob_expert_evaluate[0, v_x], ob_expert_evaluate[
                            0, v_y], \
                                                                         ob_expert_evaluate[0, a_x], ob_expert_evaluate[
                                                                             0, a_y]

                        # print('abspos_yaw_expert_evaluate: ', abspos_yaw_expert_evaluate)
                        v_x_expert_zombie_0, v_y_expert_zombie_0, a_x_expert_zombie_0, a_y_expert_zombie_0 = \
                            ob_expert_evaluate[0, zombie_0_v_x], ob_expert_evaluate[0, zombie_0_v_y], ob_expert_evaluate[0,
                                                                                                                         zombie_0_a_x], \
                            ob_expert_evaluate[0, zombie_0_a_y]
                        # print('zombie_1_v_x: ', zombie_1_v_x, 'zombie_1_v_y: ', zombie_1_v_y, 'zombie_1_a_x: ', zombie_1_a_x, 'zombie_1_a_y: ', zombie_1_a_y)
                        v_x_expert_zombie_1, v_y_expert_zombie_1, a_x_expert_zombie_1, a_y_expert_zombie_1 = \
                            ob_expert_evaluate[0, zombie_1_v_x], ob_expert_evaluate[0, zombie_1_v_y], ob_expert_evaluate[0,
                                                                                                                         zombie_1_a_x], \
                            ob_expert_evaluate[0, zombie_1_a_y]
                        box_p0_x_expert_zombie_0, box_p0_y_expert_zombie_0, box_p1_x_expert_zombie_0, box_p1_y_expert_zombie_0, box_p2_x_expert_zombie_0, box_p2_y_expert_zombie_0, box_p3_x_expert_zombie_0, box_p3_y_expert_zombie_0 = \
                            ob_expert_evaluate[0, zombie_0_box_p0_x], ob_expert_evaluate[
                                0, zombie_0_box_p0_y], ob_expert_evaluate[0,
                                                                          zombie_0_box_p1_x], ob_expert_evaluate[
                                0, zombie_0_box_p1_y], ob_expert_evaluate[0,
                                                                          zombie_0_box_p2_x], ob_expert_evaluate[
                                0, zombie_0_box_p2_y], ob_expert_evaluate[0,
                                                                          zombie_0_box_p3_x], ob_expert_evaluate[
                                0, zombie_0_box_p3_y]
                        box_p0_x_expert_zombie_1, box_p0_y_expert_zombie_1, box_p1_x_expert_zombie_1, box_p1_y_expert_zombie_1, box_p2_x_expert_zombie_1, box_p2_y_expert_zombie_1, box_p3_x_expert_zombie_1, box_p3_y_expert_zombie_1 = \
                            ob_expert_evaluate[0, zombie_1_box_p0_x], ob_expert_evaluate[
                                0, zombie_1_box_p0_y], ob_expert_evaluate[0,
                                                                          zombie_1_box_p1_x], ob_expert_evaluate[
                                0, zombie_1_box_p1_y], ob_expert_evaluate[0,
                                                                          zombie_1_box_p2_x], ob_expert_evaluate[
                                0, zombie_1_box_p2_y], ob_expert_evaluate[0,
                                                                          zombie_1_box_p3_x], ob_expert_evaluate[
                                0, zombie_1_box_p3_y]
                        abs_x_expert, abs_y_expert, abs_z_expert, yaw_expert = abspos_yaw_expert_evaluate[0, 0], \
                                                                               abspos_yaw_expert_evaluate[0, 1], \
                                                                               abspos_yaw_expert_evaluate[0, 2], \
                                                                               abspos_yaw_expert_evaluate[0, 3]

                        for i in range(len(ob)):
                            abs_x_generator, abs_y_generator, abs_z_generator, yaw_generator = current_pos[i][0], \
                                                                                               current_pos[i][1], \
                                                                                               current_pos[i][2], \
                                                                                               yaw[i]

                            if abs(abs_x_generator - abs_x_expert) < 0.5 and abs(abs_y_generator - abs_y_expert) < 0.5:
                                # print('abs_x_generator - abs_x_expert: ', abs_x_generator - abs_x_expert,
                                #       'abs_y_generator - abs_y_expert: ', abs_y_generator - abs_y_expert)
                                #
                                # print('abs_x_generator, abs_y_generator, abs_z_generator, yaw_generator: ',
                                #       abs_x_generator, abs_y_generator, abs_z_generator, yaw_generator, yaw_generator.item)
                                # print('abs_x_expert, abs_y_expert, abs_z_expert, yaw_expert: ',
                                #       abs_x_expert, abs_y_expert, abs_z_expert, yaw_expert)
                                v_x_generator, v_y_generator, a_x_generator, a_y_generator = ob[i][v_x], ob[i][v_y], ob[i][
                                    a_x], ob[i][a_y]
                                v_x_generator_zombie_0, v_y_generator_zombie_0, a_x_generator_zombie_0, a_y_generator_zombie_0 = \
                                    ob[i][zombie_0_v_x], ob[i][zombie_0_v_y], ob[i][zombie_0_a_x], ob[i][zombie_0_a_y]
                                v_x_generator_zombie_1, v_y_generator_zombie_1, a_x_generator_zombie_1, a_y_generator_zombie_1 = \
                                    ob[i][zombie_1_v_x], ob[i][zombie_1_v_y], ob[i][zombie_1_a_x], ob[i][zombie_1_a_y]
                                box_p0_x_generator_zombie_0, box_p0_y_generator_zombie_0, box_p1_x_generator_zombie_0, box_p1_y_generator_zombie_0, box_p2_x_generator_zombie_0, box_p2_y_generator_zombie_0, box_p3_x_generator_zombie_0, box_p3_y_generator_zombie_0 = \
                                    ob[i][zombie_0_box_p0_x], ob[i][zombie_0_box_p0_y], ob[i][zombie_0_box_p1_x], ob[i][
                                        zombie_0_box_p1_y], ob[i][zombie_0_box_p2_x], ob[i][zombie_0_box_p2_y], ob[i][
                                        zombie_0_box_p3_x], ob[i][zombie_0_box_p3_y]
                                box_p0_x_generator_zombie_1, box_p0_y_generator_zombie_1, box_p1_x_generator_zombie_1, box_p1_y_generator_zombie_1, box_p2_x_generator_zombie_1, box_p2_y_generator_zombie_1, box_p3_x_generator_zombie_1, box_p3_y_generator_zombie_1 = \
                                    ob[i][zombie_1_box_p0_x], ob[i][zombie_1_box_p0_y], ob[i][zombie_1_box_p1_x], ob[i][
                                        zombie_1_box_p1_y], ob[i][zombie_1_box_p2_x], ob[i][zombie_1_box_p2_y], ob[i][
                                        zombie_1_box_p3_x], ob[i][zombie_1_box_p3_y]

                                # print('v_x_generator, v_y_generator, a_x_generator, a_y_generator: ', v_x_generator, v_y_generator, a_x_generator, a_y_generator)
                                # print('v_x_generator_zombie_0, v_y_generator_zombie_0, a_x_generator_zombie_0, a_y_generator_zombie_0: ', v_x_generator_zombie_0, v_y_generator_zombie_0, a_x_generator_zombie_0, a_y_generator_zombie_0)
                                # print('v_x_generator_zombie_1, v_y_generator_zombie_1, a_x_generator_zombie_1, a_y_generator_zombie_1: ', v_x_generator_zombie_1, v_y_generator_zombie_1, a_x_generator_zombie_1, a_y_generator_zombie_1)
                                # print('box_p0_x_generator_zombie_0, box_p0_y_generator_zombie_0, box_p1_x_generator_zombie_0, box_p1_y_generator_zombie_0, box_p2_x_generator_zombie_0, box_p2_y_generator_zombie_0, box_p3_x_generator_zombie_0, box_p3_y_generator_zombie_0: ',
                                #       box_p0_x_generator_zombie_0, box_p0_y_generator_zombie_0, box_p1_x_generator_zombie_0, box_p1_y_generator_zombie_0, box_p2_x_generator_zombie_0, box_p2_y_generator_zombie_0, box_p3_x_generator_zombie_0, box_p3_y_generator_zombie_0)
                                # print('box_p0_x_generator_zombie_1, box_p0_y_generator_zombie_1, box_p1_x_generator_zombie_1, box_p1_y_generator_zombie_1, box_p2_x_generator_zombie_1, box_p2_y_generator_zombie_1, box_p3_x_generator_zombie_1, box_p3_y_generator_zombie_1: ',
                                #       box_p0_x_generator_zombie_1, box_p0_y_generator_zombie_1, box_p1_x_generator_zombie_1, box_p1_y_generator_zombie_1, box_p2_x_generator_zombie_1, box_p2_y_generator_zombie_1, box_p3_x_generator_zombie_1, box_p3_y_generator_zombie_1)

                                v_x_expert_dict[keypoint].append(v_x_expert)
                                v_y_expert_dict[keypoint].append(v_y_expert);
                                a_x_expert_dict[keypoint].append(a_x_expert);
                                a_y_expert_dict[keypoint].append(a_y_expert)
                                v_x_generator_dict[keypoint].append(v_x_generator);
                                v_y_generator_dict[keypoint].append(v_y_generator);
                                a_x_generator_dict[keypoint].append(a_x_generator);
                                a_y_generator_dict[keypoint].append(a_y_generator)
                                v_x_relative_dict[keypoint].append(v_x_expert - v_x_generator);
                                v_y_relative_dict[keypoint].append(v_y_expert - v_y_generator);
                                a_x_relative_dict[keypoint].append(a_x_expert - a_x_generator);
                                a_y_relative_dict[keypoint].append(a_y_expert - a_y_generator)

                                v_x_expert_zombie_0_dict[keypoint].append(v_x_expert_zombie_0);
                                v_y_expert_zombie_0_dict[keypoint].append(v_y_expert_zombie_0);
                                a_x_expert_zombie_0_dict[keypoint].append(a_x_expert_zombie_0);
                                a_y_expert_zombie_0_dict[keypoint].append(a_y_expert_zombie_0)
                                v_x_generator_zombie_0_dict[keypoint].append(v_x_generator_zombie_0);
                                v_y_generator_zombie_0_dict[keypoint].append(v_y_generator_zombie_0);
                                a_x_generator_zombie_0_dict[keypoint].append(a_x_generator_zombie_0);
                                a_y_generator_zombie_0_dict[keypoint].append(a_y_generator_zombie_0)
                                v_x_relative_zombie_0_dict[keypoint].append(v_x_expert_zombie_0 - v_x_generator_zombie_0);
                                v_y_relative_zombie_0_dict[keypoint].append(v_y_expert_zombie_0 - v_y_generator_zombie_0);
                                a_x_relative_zombie_0_dict[keypoint].append(a_x_expert_zombie_0 - a_x_generator_zombie_0);
                                a_y_relative_zombie_0_dict[keypoint].append(a_y_expert_zombie_0 - a_y_generator_zombie_0)

                                v_x_expert_zombie_1_dict[keypoint].append(v_x_expert_zombie_1);
                                v_y_expert_zombie_1_dict[keypoint].append(v_y_expert_zombie_1);
                                a_x_expert_zombie_1_dict[keypoint].append(a_x_expert_zombie_1);
                                a_y_expert_zombie_1_dict[keypoint].append(a_y_expert_zombie_1)
                                v_x_generator_zombie_1_dict[keypoint].append(v_x_generator_zombie_1);
                                v_y_generator_zombie_1_dict[keypoint].append(v_y_generator_zombie_1);
                                a_x_generator_zombie_1_dict[keypoint].append(a_x_generator_zombie_1);
                                a_y_generator_zombie_1_dict[keypoint].append(a_y_generator_zombie_1)
                                v_x_relative_zombie_1_dict[keypoint].append(v_x_expert_zombie_1 - v_x_generator_zombie_1);
                                v_y_relative_zombie_1_dict[keypoint].append(v_y_expert_zombie_1 - v_y_generator_zombie_1);
                                a_x_relative_zombie_1_dict[keypoint].append(a_x_expert_zombie_1 - a_x_generator_zombie_1);
                                a_y_relative_zombie_1_dict[keypoint].append(a_y_expert_zombie_1 - a_y_generator_zombie_1)

                                box_p0_x_expert_zombie_0_dict[keypoint].append(box_p0_x_expert_zombie_0);
                                box_p0_y_expert_zombie_0_dict[keypoint].append(box_p0_y_expert_zombie_0);
                                box_p1_x_expert_zombie_0_dict[keypoint].append(box_p1_x_expert_zombie_0);
                                box_p1_y_expert_zombie_0_dict[keypoint].append(box_p1_y_expert_zombie_0);
                                box_p2_x_expert_zombie_0_dict[keypoint].append(box_p2_x_expert_zombie_0);
                                box_p2_y_expert_zombie_0_dict[keypoint].append(box_p2_y_expert_zombie_0);
                                box_p3_x_expert_zombie_0_dict[keypoint].append(box_p3_x_expert_zombie_0);
                                box_p3_y_expert_zombie_0_dict[keypoint].append(box_p3_y_expert_zombie_0)
                                box_p0_x_generator_zombie_0_dict[keypoint].append(box_p0_x_generator_zombie_0);
                                box_p0_y_generator_zombie_0_dict[keypoint].append(box_p0_y_generator_zombie_0);
                                box_p1_x_generator_zombie_0_dict[keypoint].append(box_p1_x_generator_zombie_0);
                                box_p1_y_generator_zombie_0_dict[keypoint].append(box_p1_y_generator_zombie_0);
                                box_p2_x_generator_zombie_0_dict[keypoint].append(box_p2_x_generator_zombie_0);
                                box_p2_y_generator_zombie_0_dict[keypoint].append(box_p2_y_generator_zombie_0);
                                box_p3_x_generator_zombie_0_dict[keypoint].append(box_p3_x_generator_zombie_0);
                                box_p3_y_generator_zombie_0_dict[keypoint].append(box_p3_y_generator_zombie_0)
                                box_p0_x_relative_zombie_0_dict[keypoint].append(
                                    box_p0_x_expert_zombie_0 - box_p0_x_generator_zombie_0);
                                box_p0_y_relative_zombie_0_dict[keypoint].append(
                                    box_p0_y_expert_zombie_0 - box_p0_y_generator_zombie_0);
                                box_p1_x_relative_zombie_0_dict[keypoint].append(
                                    box_p1_x_expert_zombie_0 - box_p1_x_generator_zombie_0);
                                box_p1_y_relative_zombie_0_dict[keypoint].append(
                                    box_p1_y_expert_zombie_0 - box_p1_y_generator_zombie_0);
                                box_p2_x_relative_zombie_0_dict[keypoint].append(
                                    box_p2_x_expert_zombie_0 - box_p2_x_generator_zombie_0);
                                box_p2_y_relative_zombie_0_dict[keypoint].append(
                                    box_p2_y_expert_zombie_0 - box_p2_y_generator_zombie_0);
                                box_p3_x_relative_zombie_0_dict[keypoint].append(
                                    box_p3_x_expert_zombie_0 - box_p3_x_generator_zombie_0);
                                box_p3_y_relative_zombie_0_dict[keypoint].append(
                                    box_p3_y_expert_zombie_0 - box_p3_y_generator_zombie_0)

                                box_p0_x_expert_zombie_1_dict[keypoint].append(box_p0_x_expert_zombie_1);
                                box_p0_y_expert_zombie_1_dict[keypoint].append(box_p0_y_expert_zombie_1);
                                box_p1_x_expert_zombie_1_dict[keypoint].append(box_p1_x_expert_zombie_1);
                                box_p1_y_expert_zombie_1_dict[keypoint].append(box_p1_y_expert_zombie_1);
                                box_p2_x_expert_zombie_1_dict[keypoint].append(box_p2_x_expert_zombie_1);
                                box_p2_y_expert_zombie_1_dict[keypoint].append(box_p2_y_expert_zombie_1);
                                box_p3_x_expert_zombie_1_dict[keypoint].append(box_p3_x_expert_zombie_1);
                                box_p3_y_expert_zombie_1_dict[keypoint].append(box_p3_y_expert_zombie_1)
                                box_p0_x_generator_zombie_1_dict[keypoint].append(box_p0_x_generator_zombie_1);
                                box_p0_y_generator_zombie_1_dict[keypoint].append(box_p0_y_generator_zombie_1);
                                box_p1_x_generator_zombie_1_dict[keypoint].append(box_p1_x_generator_zombie_1);
                                box_p1_y_generator_zombie_1_dict[keypoint].append(box_p1_y_generator_zombie_1);
                                box_p2_x_generator_zombie_1_dict[keypoint].append(box_p2_x_generator_zombie_1);
                                box_p2_y_generator_zombie_1_dict[keypoint].append(box_p2_y_generator_zombie_1);
                                box_p3_x_generator_zombie_1_dict[keypoint].append(box_p3_x_generator_zombie_1);
                                box_p3_y_generator_zombie_1_dict[keypoint].append(box_p3_y_generator_zombie_1)
                                box_p0_x_relative_zombie_1_dict[keypoint].append(
                                    box_p0_x_expert_zombie_1 - box_p0_x_generator_zombie_1);
                                box_p0_y_relative_zombie_1_dict[keypoint].append(
                                    box_p0_y_expert_zombie_1 - box_p0_y_generator_zombie_1);
                                box_p1_x_relative_zombie_1_dict[keypoint].append(
                                    box_p1_x_expert_zombie_1 - box_p1_x_generator_zombie_1);
                                box_p1_y_relative_zombie_1_dict[keypoint].append(
                                    box_p1_y_expert_zombie_1 - box_p1_y_generator_zombie_1);
                                box_p2_x_relative_zombie_1_dict[keypoint].append(
                                    box_p2_x_expert_zombie_1 - box_p2_x_generator_zombie_1);
                                box_p2_y_relative_zombie_1_dict[keypoint].append(
                                    box_p2_y_expert_zombie_1 - box_p2_y_generator_zombie_1);
                                box_p3_x_relative_zombie_1_dict[keypoint].append(
                                    box_p3_x_expert_zombie_1 - box_p3_x_generator_zombie_1);
                                box_p3_y_relative_zombie_1_dict[keypoint].append(
                                    box_p3_y_expert_zombie_1 - box_p3_y_generator_zombie_1)

                                abs_x_expert_dict[keypoint].append(abs_x_expert)
                                abs_y_expert_dict[keypoint].append(abs_y_expert);
                                abs_z_expert_dict[keypoint].append(abs_z_expert);
                                yaw_expert_dict[keypoint].append(yaw_expert)
                                abs_x_generator_dict[keypoint].append(abs_x_generator)
                                abs_y_generator_dict[keypoint].append(abs_y_generator);
                                abs_z_generator_dict[keypoint].append(abs_z_generator);
                                yaw_generator_dict[keypoint].append(yaw_generator)

                                abs_x_relative_dict[keypoint].append(abs_x_expert - abs_x_generator);
                                abs_y_relative_dict[keypoint].append(abs_y_expert - abs_y_generator);
                                abs_z_relative_dict[keypoint].append(abs_z_expert - abs_z_generator);
                                yaw_relative_dict[keypoint].append(yaw_expert - yaw_generator)

                                if len(v_x_relative_dict[keypoint]) > statistic_len:
                                    v_x_relative_dict[keypoint].pop(0)
                                if len(v_y_relative_dict[keypoint]) > statistic_len:
                                    v_y_relative_dict[keypoint].pop(0)
                                if len(a_x_relative_dict[keypoint]) > statistic_len:
                                    a_x_relative_dict[keypoint].pop(0)
                                if len(a_y_relative_dict[keypoint]) > statistic_len:
                                    a_y_relative_dict[keypoint].pop(0)
                                if len(v_x_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    v_x_relative_zombie_0_dict[keypoint].pop(0)
                                if len(v_y_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    v_y_relative_zombie_0_dict[keypoint].pop(0)
                                if len(a_x_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    a_x_relative_zombie_0_dict[keypoint].pop(0)
                                if len(a_y_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    a_y_relative_zombie_0_dict[keypoint].pop(0)

                                if len(v_x_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    v_x_relative_zombie_1_dict[keypoint].pop(0)
                                if len(v_y_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    v_y_relative_zombie_1_dict[keypoint].pop(0)
                                if len(a_x_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    a_x_relative_zombie_1_dict[keypoint].pop(0)
                                if len(a_y_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    a_y_relative_zombie_1_dict[keypoint].pop(0)

                                if len(box_p0_x_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    box_p0_x_relative_zombie_0_dict[keypoint].pop(0)
                                if len(box_p1_x_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    box_p1_x_relative_zombie_0_dict[keypoint].pop(0)
                                if len(box_p2_x_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    box_p2_x_relative_zombie_0_dict[keypoint].pop(0)
                                if len(box_p3_x_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    box_p3_x_relative_zombie_0_dict[keypoint].pop(0)
                                if len(box_p0_y_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    box_p0_y_relative_zombie_0_dict[keypoint].pop(0)
                                if len(box_p1_y_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    box_p1_y_relative_zombie_0_dict[keypoint].pop(0)
                                if len(box_p2_y_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    box_p2_y_relative_zombie_0_dict[keypoint].pop(0)
                                if len(box_p3_y_relative_zombie_0_dict[keypoint]) > statistic_len:
                                    box_p3_y_relative_zombie_0_dict[keypoint].pop(0)

                                if len(box_p0_x_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    box_p0_x_relative_zombie_1_dict[keypoint].pop(0)
                                if len(box_p1_x_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    box_p1_x_relative_zombie_1_dict[keypoint].pop(0)
                                if len(box_p2_x_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    box_p2_x_relative_zombie_1_dict[keypoint].pop(0)
                                if len(box_p3_x_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    box_p3_x_relative_zombie_1_dict[keypoint].pop(0)
                                if len(box_p0_y_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    box_p0_y_relative_zombie_1_dict[keypoint].pop(0)
                                if len(box_p1_y_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    box_p1_y_relative_zombie_1_dict[keypoint].pop(0)
                                if len(box_p2_y_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    box_p2_y_relative_zombie_1_dict[keypoint].pop(0)
                                if len(box_p3_y_relative_zombie_1_dict[keypoint]) > statistic_len:
                                    box_p3_y_relative_zombie_1_dict[keypoint].pop(0)
                            else:
                                pass
                        # try:
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_x_ego_mean",
                                              np.mean(v_x_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_x_ego_variance",
                                              np.var(v_x_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_y_ego_mean",
                                              np.mean(v_y_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_y_ego_variance",
                                              np.var(v_y_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_x_ego_mean",
                                              np.mean(a_x_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_x_ego_variance",
                                              np.var(a_x_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_y_ego_mean",
                                              np.mean(a_y_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_y_ego_variance",
                                              np.var(a_y_relative_dict[keypoint]))

                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_x_zombie_0_mean",
                                              np.mean(v_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_x_zombie_0_variance",
                                              np.var(v_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_y_zombie_0_mean",
                                              np.mean(v_y_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_y_zombie_0_variance",
                                              np.var(v_y_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_x_zombie_0_mean",
                                              np.mean(a_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_x_zombie_0_variance",
                                              np.var(a_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_y_zombie_0_mean",
                                              np.mean(a_y_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_y_zombie_0_variance",
                                              np.var(a_y_relative_zombie_0_dict[keypoint]))

                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_x_zombie_1_mean",
                                              np.mean(v_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_x_zombie_1_variance",
                                              np.var(v_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_y_zombie_1_mean",
                                              np.mean(v_y_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "v_y_zombie_1_variance",
                                              np.var(v_y_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_x_zombie_1_mean",
                                              np.mean(a_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_x_zombie_1_variance",
                                              np.var(a_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_y_zombie_1_mean",
                                              np.mean(a_y_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "a_y_zombie_1_variance",
                                              np.var(a_y_relative_zombie_1_dict[keypoint]))

                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p0_x_zombie_0_mean",
                                              np.mean(box_p0_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p0_x_zombie_0_variance",
                                              np.var(box_p0_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p1_x_zombie_0_mean",
                                              np.mean(box_p1_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p1_x_zombie_0_variance",
                                              np.var(box_p1_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p2_x_zombie_0_mean",
                                              np.mean(box_p2_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p2_x_zombie_0_variance",
                                              np.var(box_p2_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p3_x_zombie_0_mean",
                                              np.mean(box_p3_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p3_x_zombie_0_variance",
                                              np.var(box_p3_x_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p0_y_zombie_0_mean",
                                              np.mean(box_p0_y_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p0_y_zombie_0_variance",
                                              np.var(box_p0_y_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p1_y_zombie_0_mean",
                                              np.mean(box_p1_y_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p1_y_zombie_0_variance",
                                              np.var(box_p1_y_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p2_y_zombie_0_mean",
                                              np.mean(box_p2_y_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p2_y_zombie_0_variance",
                                              np.var(box_p2_y_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p3_y_zombie_0_mean",
                                              np.mean(box_p3_y_relative_zombie_0_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p3_y_zombie_0_variance",
                                              np.var(box_p3_y_relative_zombie_0_dict[keypoint]))

                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p0_x_zombie_1_mean",
                                              np.mean(box_p0_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p0_x_zombie_1_variance",
                                              np.var(box_p0_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p1_x_zombie_1_mean",
                                              np.mean(box_p1_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p1_x_zombie_1_variance",
                                              np.var(box_p1_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p2_x_zombie_1_mean",
                                              np.mean(box_p2_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p2_x_zombie_1_variance",
                                              np.var(box_p2_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p3_x_zombie_1_mean",
                                              np.mean(box_p3_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p3_x_zombie_1_variance",
                                              np.var(box_p3_x_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p0_y_zombie_1_mean",
                                              np.mean(box_p0_y_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p0_y_zombie_1_variance",
                                              np.var(box_p0_y_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p1_y_zombie_1_mean",
                                              np.mean(box_p1_y_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p1_y_zombie_1_variance",
                                              np.var(box_p1_y_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p2_y_zombie_1_mean",
                                              np.mean(box_p2_y_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p2_y_zombie_1_variance",
                                              np.var(box_p2_y_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p3_y_zombie_1_mean",
                                              np.mean(box_p3_y_relative_zombie_1_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "box_p3_y_zombie_1_variance",
                                              np.var(box_p3_y_relative_zombie_1_dict[keypoint]))

                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "abs_x_mean",
                                              np.mean(abs_x_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "abs_x_variance",
                                              np.var(abs_x_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "abs_y_mean",
                                              np.mean(abs_y_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "abs_y_variance",
                                              np.var(abs_y_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "abs_z_mean",
                                              np.mean(abs_z_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "abs_z_variance",
                                              np.var(abs_z_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "yaw_mean",
                                              np.mean(yaw_relative_dict[keypoint]))
                        logger.record_tabular(stage + "_" + str(keypoint) + "_Feature_" + "yaw_variance",
                                              np.var(yaw_relative_dict[keypoint]))
                        # except:
                        #     print('Logger passed!')

                        steer_, throttle_ = 0, 1
                        fakerew_standard = self.reward_giver.get_reward(ob_expert_evaluate, ac_expert_evaluate)[0][0]
                        v_x_expert, v_y_expert, a_x_expert, a_y_expert = ob_expert_evaluate[0][v_x], ob_expert_evaluate[0][
                            v_y], ob_expert_evaluate[0][a_x], ob_expert_evaluate[0][a_y]
                        steer_expert, throttle_expert = ac_expert_evaluate[0][steer_], ac_expert_evaluate[0][throttle_]
                        prefix = outstr(v_x_expert, v_y_expert, a_x_expert, a_y_expert, steer_expert, throttle_expert)
                        # prefix = stage +"_AT_"+prefix
                        prefix = stage + "_" + str(keypoint) + "_T" + prefix
                        logger.record_tabular(prefix, fakerew_standard)

                        ob_indexes = {"Vx": v_x, "Vy": v_y, "Ax": a_x, "Ay": a_y}
                        ac_indexes = {"St": steer_, "Th": throttle_}

                        indexes = [ob_indexes, ac_indexes]
                        for i in range(len(indexes)):
                            if i == 0:
                                index = indexes[i]
                                ac_ctfl = ac_expert_evaluate
                                for k, idx in index.items():
                                    ob_ctfl = copy(ob_expert_evaluate)
                                    less_value = ob_ctfl[0][idx] / 2.
                                    ob_ctfl[0][idx] = less_value
                                    prefix_ctfl = stage + "_" + str(keypoint) + "_FakeRew_" + k + str(round(less_value, 1))
                                    fakerew_ctfl = self.reward_giver.get_reward(ob_ctfl, ac_ctfl)[0][0]
                                    # logger.record_tabular(prefix_ctfl, fakerew_ctfl)
                                    #relative_rew = abs(fakerew_ctfl - fakerew_standard)
                                    relative_rew = fakerew_standard - fakerew_ctfl
                                    logger.record_tabular(prefix_ctfl, relative_rew)

                                    ob_ctfl = copy(ob_expert_evaluate)
                                    negative_value = ob_ctfl[0][idx] * -1.
                                    ob_ctfl[0][idx] = negative_value
                                    prefix_ctfl = stage + "_" + str(keypoint) + "_FakeRew_" + k + str(round(negative_value, 1))
                                    fakerew_ctfl = self.reward_giver.get_reward(ob_ctfl, ac_ctfl)[0][0]
                                    # logger.record_tabular(prefix_ctfl, fakerew_ctfl)
                                    #relative_rew = abs(fakerew_ctfl - fakerew_standard)
                                    relative_rew = fakerew_standard - fakerew_ctfl
                                    logger.record_tabular(prefix_ctfl, relative_rew)

                            elif i == 1:
                                index = indexes[i]
                                ob_ctfl = ob_expert_evaluate
                                for k, idx in index.items():
                                    ac_ctfl = copy(ac_expert_evaluate)
                                    less_value = ac_ctfl[0][idx] / 2.
                                    ac_ctfl[0][idx] = less_value
                                    prefix_ctfl = stage + "_" + str(keypoint) + "_FakeRew_" + k + str(round(less_value, 1))
                                    fakerew_ctfl = self.reward_giver.get_reward(ob_ctfl, ac_ctfl)[0][0]
                                    # logger.record_tabular(prefix_ctfl, fakerew_ctfl)
                                    #relative_rew = abs(fakerew_ctfl - fakerew_standard)
                                    relative_rew = fakerew_standard - fakerew_ctfl
                                    logger.record_tabular(prefix_ctfl, relative_rew)

                                    if k == "St":
                                        ac_ctfl = copy(ac_expert_evaluate)
                                        negative_value = ac_ctfl[0][idx] * -1.
                                        ac_ctfl[0][idx] = negative_value
                                        prefix_ctfl = stage + "_" + str(keypoint) + "_FakeRew_" + k + str(round(negative_value, 1))
                                        fakerew_ctfl = self.reward_giver.get_reward(ob_ctfl, ac_ctfl)[0][0]
                                        # logger.record_tabular(prefix_ctfl, fakerew_ctfl)
                                        #relative_rew = abs(fakerew_ctfl - fakerew_standard)
                                        relative_rew = fakerew_standard - fakerew_ctfl
                                        logger.record_tabular(prefix_ctfl, relative_rew)

            if self.rank == 0:
                logger.dump_tabular()

        pi_params = [tf.get_default_session().run(param) for param in self.pi.get_variables()]
        dis_params = [tf.get_default_session().run(param) for param in self.reward_giver.get_trainable_variables()]
        return [pi_params, dis_params]

    def get_params(self):
        pi_params = [tf.get_default_session().run(param) for param in self.pi.get_variables()]
        dis_params = [tf.get_default_session().run(param) for param in self.reward_giver.get_trainable_variables()]
        return [pi_params, dis_params]

    def load_model(self):
        logger.log("Let's load the pretrained model")
        logger.log(self.args.load_model_path)
        saver = tf.train.Saver(max_to_keep=5000)
        sess = tf.get_default_session()
        params = sess.run(self.pi.get_trainable_variables())
        saver.restore(tf.get_default_session(), self.args.load_model_path)
        params = sess.run(self.pi.get_trainable_variables())
    
    def load_dmodel(self):
        logger.log("Let's load the pretrained discriminator model")
        logger.log(self.args.load_model_path)
        saver = tf.train.Saver(var_list=self.reward_giver.get_trainable_variables(), max_to_keep=5000)
        sess = tf.get_default_session()
        saver.restore(tf.get_default_session(), self.args.load_model_path)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

@ray.remote
def actor_worker(*actors):
    batches = ray.get([actor.get_batch.remote() for actor in actors])
    return batches

def stack(array):
    if len(array.shape) == 2:
        array = array.reshape(np.prod(array.shape))
    elif len(array.shape) == 3:
        array = array.reshape(np.prod(array.shape[:-1]), -1)
    elif len(array.shape) == 1:
        pass
    return array

def batches_filter(batches):
    batches_original = batches

    keys = batches[0].keys()
    keys = sorted(keys)
    ep_infos = ["ep_rets", "ep_lens", "ep_true_rets", 'ep_v', 'ep_acc', 'ep_left_offset', 'ep_right_offset', "ep_true_lens"]
    ep_flat = ["v_ep", "scene", "ep_rets"]
    ep_keys = ep_infos + ep_flat
    keys = [key for key in keys if key not in ep_keys]
    batches_data = [np.array([batch[key] for batch in batches]) for key in keys]
    batches_data = [stack(data) for data in batches_data]
    batches = {keys[i]:batches_data[i] for i in range(len(keys))}

    def safemean(xs):
        return np.nan if len(xs) == 0 else np.mean(xs)
    def safemax(xs):
        return np.nan if len(xs) == 0 else np.max(xs)

    ep_scala_keys = ['ep_acc', 'ep_v', 'ep_left_offset', 'ep_right_offset']
    for key in ep_scala_keys:
        batches[key] = safemean([batch[key] for batch in batches_original])
    for key in [_key for _key in ep_keys if _key not in ep_scala_keys]:
        output = []
        for batch in batches_original:
            output += batch[key]
        batches[key] = output
    return batches

def merge(expert_dataset):
    obs = []
    acs = []
    obs_abspos_yaw = []
    for e_data in expert_dataset:
        for traj in e_data.obs:
            obs.append(traj)
        for traj in e_data.acs:
            acs.append(traj)
        for traj in e_data.obs_abspos_yaw:
            obs_abspos_yaw.append(traj)
    obs = np.array(obs)
    acs = np.array(acs)
    obs_abspos_yaw = np.array(obs_abspos_yaw)
    expert_dataset_origin = copy(expert_dataset)
    expert_dataset = expert_dataset[0]
    expert_dataset.obs, expert_dataset.acs, expert_dataset.obs_abspos_yaw = obs, acs, obs_abspos_yaw
    print("==========================================================================")
    print("Expert Data for each scenario: ", expert_dataset_origin[0].obs.shape)
    print("Merge Expert Data: ", expert_dataset.obs.shape)
    print("==========================================================================")
    return expert_dataset

def main(args):
    ray.init(num_gpus=args.actor_nums+1)
    host = [str(x) for x in args.host.split('_') if x!='']
    port = [int(x) for x in args.port.split('_') if x!='']

    split = int(len(host)/args.actor_nums)
    nums = int(len(host))
    hosts = [host[i:i+split] for i in range(0, nums, split)]
    ports = [port[i:i+split] for i in range(0, nums, split)]
    print(hosts, ports)
    args.hosts = hosts
    args.ports = ports

    actors = [Actor.remote(args, i) for i in range(args.actor_nums)]
    
    if args.scenario_name == "Merge_Env" or args.speed_mode == "mix":
        args = [ray.get(actors[i].get_args.remote()) for i in range(len(actors))]
        expert_dataset = [args[i].expert_dataset for i in range(len(actors))]
        expert_dataset = merge(expert_dataset)
        args[0].expert_dataset = expert_dataset
        args = args[0]
    else:
        args = ray.get(actors[0].get_args.remote())

    learner = Learner.remote(args)
    # init actor parameters
    params = ray.get(learner.get_params.remote())
    status = [actor.update_params.remote(params) for actor in actors]

    g_steps = 0
    iteration = 0
    logger.log("********** Iteration %i ************" % iteration)
    if args.task == 'generate_data':
        save_file = open("./tmp_TRPO_curriculum.pkl", "wb")
    while True:
        batches = ray.get([actor.get_batch.remote() for actor in actors])
        batches = batches_filter(batches)
        if args.task == 'generate_data':
            ob_batch = batches['ob'].tolist()
            control_batch = batches['ctrl'].tolist()
            current_pos_batch = batches['current_pos'].tolist()
            yaw_batch = batches['yaw'].tolist()
            print(len(ob_batch), len(ob_batch[0]), len(current_pos_batch[0]), yaw_batch[0])
            assert len(ob_batch) == len(control_batch) == len(current_pos_batch) == len(yaw_batch), 'len(ob_batch) != len(control_batch)'
            for i in range(len(ob_batch)):
                pickle.dump(ob_batch[i] + current_pos_batch[i] + yaw_batch[i] + control_batch[i], save_file)

        if args.task == 'train':
            params = ray.get(learner.learn.remote(batches))
            status = [actor.update_params.remote(params) for actor in actors]
        g_steps += 1
        if g_steps % args.g_step == 0:
            iteration += 1
            logger.log("********** Iteration %i ************" % iteration)
        if iteration > args.max_iters:
            break 

    if args.task == 'generate_data':
        save_file.close()

    return

if __name__ == '__main__':
    args = argsparser()
    main(args)

