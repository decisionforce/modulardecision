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

import glob, os
import sys
from copy import copy

current_dir = os.path.dirname(__file__)
sys.path.insert(0,os.path.dirname(os.path.dirname(current_dir)))

# add carla config here, put carla directory as the parent directory of IDM
carla_simulator_path = '../'
carla_path = carla_simulator_path + 'PythonAPI/carla-0.9.1-py3.5-linux-x86_64.egg' 
try: 
    sys.path.append(carla_path)
except IndexError:
    pass

from baselines.gail.gail_control import Carla, carla, pygame, HUD, World
from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.dataset.mujoco_dset import Carla_Dset, Mujoco_Dset
from baselines.gail.adversary import TransitionClassifier
from baselines.logger import TensorBoardOutputFormat
import subprocess
import time

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Carla-Motion')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='../log/easy.pkl')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    # parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default='False')
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample', 'bc_evaluate'], default='train')
    # for evaluatation
    # boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    # boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--stochastic_policy', help='use stochastic/deterministic policy to evaluate', type=str, default='False')
    parser.add_argument('--save_sample', help='save the trajectories or not', type=str, default='False')

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
    parser.add_argument('--d_actv', type=str, default="relu", help='Activation for discriminator, default is relu')
    parser.add_argument('--sn', action='store_false', help='Spectral normalization on Discriminator')
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    parser.add_argument('--d_lrate', type=float, default=0.001)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=80)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    # boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--pretrained', type=str, default="False", help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    # Carla settings
    parser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    parser.add_argument('--render', action='store_true', help='Render botton')
    parser.add_argument('--num_trajectories', metavar='T', default=100, type=int, help='num of trajectories')
    parser.add_argument('--num_length', metavar='L', default=200, type=int, help='length of one trajectory')
    parser.add_argument('--batch_size', default=1024, type=int, help='timesteps per batch')
    parser.add_argument('--search', action='store_true', help='search for nearest expert path for training')
    parser.add_argument('--stack', default=1, type=int, help='stack frames')
    parser.add_argument('--search_mode', default='traj', type=str, help='search mode, default is nearest trajectory')
    parser.add_argument('--scene', default='all', type=str, choices=['all', 'straight', 'curve'], help='training scene')
    parser.add_argument('--mode', default='all', type=str, choices=['all', 'wp', 'wp_obj'], help='observation dim')
    parser.add_argument('--train_mode', default='all', type=str, choices=['all', 'steer'], help='choose which action to train')
    parser.add_argument('--feature', default='wp_car', type=str, choices=['wp_car', 'wp_all'], help='features')
    parser.add_argument('--d_model', default='origin', type=str, choices=['origin', "separate"], help='model for discriminator')
    parser.add_argument('--skip', default=2, type=int, choices=[1, 2, 4, 8], help='skip frames')
    parser.add_argument('--p_update', action='store_false', help='policy update')
    parser.add_argument('--rew_type', action='store_true', help='true reward to update')
    parser.add_argument('--detect', action='store_true', help='whether to detect right or wrong')
    parser.add_argument('--dim', default='2d', type=str, choices=['2d', '3d'], help='observations dimension')
    parser.add_argument('--region', default=0.2, type=float, choices=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2], help='region for early reset')
    parser.add_argument('--resampling', default=0, type=int, choices=[0, 2, 4, 8, 12], help='resampling for increasing observation diversity')
    parser.add_argument('--gamma', default=0.995, type=float, help='gamma discounting factor')
    parser.add_argument('--dis_max', default=1.35, type=float, help='longest distance for lane sampling')
    parser.add_argument('--r_norm', action='store_true', help='reward normalization')
    parser.add_argument('--init_std', default=1., type=float, help='initial std')
    parser.add_argument('--carla_path', default='../', type=str, help='relative path of the folder of carlaUE4.sh')
    parser.add_argument('--spawn_mode', default='random', type=str, choices=['fixed', 'random'], help='spawn mode')

    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    return args


def get_task_name(args):
    time_pre = time.strftime("%Y%m%d%H%M%S", time.localtime())
    if args.rew_type:
        task_name = time_pre+'_'+args.algo
    else:
        task_name = time_pre+'_'+args.algo+ "_GAIL."
    if args.r_norm:
        task_name += 'Norm'
    if args.pretrained:
        task_name += "PreT%s." % args.BC_max_iter
    if args.search:
        task_name += "_Search%s_" % args.search_mode
    if args.sn:
        task_name += "SN_"
    if not args.p_update:
        task_name += 'NoP'
    task_name += '%s_' % args.spawn_mode
    task_name += 'STD%s_'% args.init_std
    PLayer = args.policy_hidden_layer * 2 + 1
    task_name += 'D%s' % args.d_model
    task_name += 'Traj%s_' % args.num_trajectories
    task_name += 'Len%s_' % args.num_length
    task_name += 'Reg%s_' % args.region
    task_name += 'MaxD%s_' % args.dis_max
    if args.search:
        task_name += 'Lim%s_' % args.traj_limitation
    task_name += 'GS%s_' % args.policy_hidden_size
    task_name += 'PL%s_' % PLayer
    task_name += 'DS%s_' % args.adversary_hidden_size
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
    task_name += 'Stack%s_' % args.stack
    task_name += 'Mode%s|'%args.mode
    task_name += '%s|'%args.train_mode
    task_name += '%s_' %args.scene
    task_name += 'Ski%s_'%args.skip
    task_name += 'Fe%s_'%args.feature
    task_name += '%s'%args.dim
    task_name += 'G%s'%args.gamma

    task_name = task_name + ".G_" + str(args.g_step) + ".D_" + str(args.d_step) + \
        ".G_entcoeff_" + str(args.policy_entcoeff) + ".D_entcoeff_" + str(args.adversary_entcoeff) + \
        ".maxkl_" + str(args.max_kl)
    task_name += ".seed_" + str(args.seed)
    return task_name

class CarlaUE4Monitor(object):
    def __init__(self,args):
        self.args = args

    def _init_server(self):
        carla_cmd = os.path.join(self.args.carla_path+'CarlaUE4.sh')
        carla_arg = ' -world-port=2000 --carla-server -fps 20 -carla-settings=./CarlaSettings.ini -windowed -ResX=104 -ResY=104 -opengl3'
        carla_cmd += carla_arg
        self.server_process = subprocess.Popen(carla_cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.server_process.wait()
        logger.info(slef.server_process.pid)
        logger.info('server launch return code:%s'%self.server_process.poll())
        if self.server_process.poll()!=0:
            raise ValueError('Carla Server launching failed')
        logger.info(carla_cmd)

    def _close_server(self):
        try:
            os.kill(self.server_process.pid+1,signal.SIGKILL)
        except OSError:
            logger.info('Process does not exist')

    def monitor(self):
        raise NotImplemented

def recover_args(args):
    if args.load_model_path == 'False':
        args.load_model_path = None
    if args.stochastic_policy == 'False':
        args.stochastic_policy = False
    else:
        args.stochastic_policy = True
    if args.save_sample == 'False':
        args.save_sample = False
    else:
        args.save_sample = True
    if args.pretrained == 'False':
        args.pretrained = False
    else:
        args.pretrained = True
    return args

def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    
    args = recover_args(args)

    # open CarlaUE4 backend
    # carlaue4_monitor = CarlaUE4Monitor(args)
    # carlaue4_monitor._init_server()
    # wait for CarlaUE4 initialization
    # time.sleep(60)

    # start carla server
    if args.env_id == "Carla-Motion":
        pygame.init()
        pygame.font.init()
        world = None

        client = carla.Client(args.host, args.port)
        client.set_timeout(600.0)
        display = None
        if args.render:
            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(carla_world=client.get_world(), hud=hud, camera=args.render, mode=args.mode,
                      feature=args.feature, dim=args.dim, dis_max=args.dis_max, spawn_mode=args.spawn_mode)
        args.dis_max = int(args.dis_max**16)
        clock = pygame.time.Clock()

        if args.task == "evaluate":
            test = True
        else:
            test = False

        env = Carla(world, clock, display, args.render, args.num_length, args.stack, args.skip, args.train_mode, 
                    test, args.region)
    else:
        env = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=args.policy_hidden_layer)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)

    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    logger.init_std = args.init_std

    if args.task == 'train':
        datashape = env.observation_space.shape[0]
        if args.env_id == "Carla-Motion":
            dataset = Carla_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation,
            num_trajectories=args.num_trajectories, num_length=args.num_length, data_shape=datashape,
            search=args.search, mode=args.mode, feature=args.feature, train_mode=args.train_mode, dim=args.dim,
            resampling=args.resampling)
        else:
            dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        logger.ckpt_dir_name = args.checkpoint_dir
        logger.configure(args.checkpoint_dir, format_strs=['stdout', 'json', 'log', 'tensorboard'])
        tfboard = TensorBoardOutputFormat(args.checkpoint_dir)
        logger.tfboard = tfboard
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff,
        adversary_hidden_layer=args.adversary_hidden_layer, sn=args.sn, d_actv=args.d_actv, lr_rate=args.d_lrate,
        model=args.d_model, train_mode=args.train_mode, dim=args.dim)
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
              )
    elif args.task == 'evaluate':
        datashape = env.observation_space.shape[0]
        logger.ckpt_dir_name = args.checkpoint_dir
        logger.configure(args.checkpoint_dir)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff, \
        adversary_hidden_layer=args.adversary_hidden_layer, sn=args.sn, d_actv=args.d_actv, lr_rate=args.d_lrate, \
        model=args.d_model, detect=args.detect, train_mode=args.train_mode)
        dataset = Carla_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation, \
        num_trajectories=args.num_trajectories, num_length=args.num_length, data_shape=datashape, \
        search=args.search, mode=args.mode, feature=args.feature)
        runner(env,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=1024,
               number_trajs=40,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample,
               reward_giver=reward_giver,
               detect=args.detect,
               dataset=dataset
               )

    elif args.task == 'bc_evaluate':
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        datashape = env.observation_space.shape[0] + env.action_space.shape[0]
        dataset = Carla_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation, \
        num_trajectories=args.num_trajectories, num_length=args.num_length, data_shape=datashape, search=args.search)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
        for i in range(100, int(args.BC_max_iter), 100):
            from baselines.gail import behavior_clone, trpo_mpi
            pretrained_weight = behavior_clone.learn(env, policy_fn, dataset, max_iters=i, idx=i)
            runner_bc(env,
                   policy_fn,
                   args.load_model_path,
                   timesteps_per_batch=1024,
                   number_trajs=10,
                   stochastic_policy=args.stochastic_policy,
                   save=args.save_sample,
                   pretrained_weight=pretrained_weight,
                   idx=i
                   )
    else:
        raise NotImplementedError
    env.close()
    # shutdown carlaue4 backend
    # carlaue4_monitor._close_server()


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, task_name=None, load_model_path=None,
          batch_size=1024, search=False, search_mode='traj', scene='all',
          p_update=True, rew_type=False, train_mode='all', gamma=0.995, r_norm=False
          ):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
                                                 max_iters=BC_max_iter)

    if algo == 'trpo':
        from baselines.gail import trpo_mpi
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=batch_size,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=gamma, lam=0.97,
                       vf_iters=5, vf_stepsize=1e-3, task_name=task_name, 
                       load_model_path=load_model_path, search=search, search_mode=search_mode,
                       scene=scene,p_update=p_update,rew_type=rew_type,train_mode=train_mode,
                       r_norm=r_norm)
    elif algo=="ppo":
        from baselines.ppo2.ppo2 import learn
        from baselines.ppo2.microbatched_model import MicrobatchedModel
        from functools import partial
        rank = MPI.COMM_WORLD.Get_rank()
        if rank!= 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        eval_env = None
        if load_model_path != None:
            eval_env = copy(env)
        learn_fn = partial(learn, network='mlp', nsteps=batch_size, nminibatches=32, lam=0.95, gamma=0.99, \
                        noptepochs=10, log_interval=1, ent_coef=0.0, lr=lambda f: 3e-4 * f, cliprange=0.2, \
                        value_network='copy', total_timesteps=num_timesteps, seed=0, start_steps=start_steps,\
                        save_interval=2, max_iters=max_iters, load_path=load_model_path, eval_env=eval_env, \
                        start_v=start_v)
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
    
        if detect:
            ob_batch = obs
            ac_batch = acs
            ob_expert, ac_expert = dataset.get_next_batch(len(ob_batch))
            error_g, g_total_number = reward_giver.detect_error(ob_batch, ac_batch, "g")
            error_e, e_total_number = reward_giver.detect_error(ob_expert, ac_expert, "e")
            print('Error generator: %f, total %f, error rate %f' % (error_g, g_total_number, float(error_g/g_total_number)))
            print('Error expert: %f, total %f, error rate %f' % (error_e, e_total_number, float(error_e/e_total_number)))

    if stochastic_policy:   
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret

def runner_bc(env, policy_func, load_model_path, timesteps_per_batch, number_trajs, 
        stochastic_policy, save=False, reuse=False, pretrained_weight=None, idx=0):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi"+str(idx), ob_space, ac_space, reuse=True)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    import tensorflow as tf
    U.load_variables(pretrained_weight, variables=pi.get_variables(), sess=tf.get_default_session())
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
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret

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

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)
        ob, rew, new, _ = env.step(ac)
        env.render()
        rews.append(rew)

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
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
