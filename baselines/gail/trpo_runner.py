'''
Disclaimer: The trpo part highly rely on trpo_mpi at @openai/baselines
'''

import time
import os
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset, fmt_row
from baselines import logger
from baselines.common import colorize
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.gail.statistics import stats
from tensorflow.contrib.tensorboard.plugins import projector
from numpy import linalg as LA
from copy import copy
from gym import spaces

def stack(array):
    if len(array.shape) == 2:
        array = array.reshape(np.prod(array.shape))
    else:
        array = array.reshape(np.prod(array.shape[:-1]), -1)
    return array

def detect_scene(scenes, ep_lens):
    _eps = len(ep_lens)
    curves = 0
    _range = []
    start_idx = 0
    for _len in ep_lens:
        _range.append((start_idx, start_idx+_len))
        start_idx = start_idx + _len
    for start_idx, end_idx in _range:
        scenes = list(scenes)
        scene_ep = scenes[start_idx: end_idx]
        for scene in range(len(scenes)):
            if scene == 'curve':
                curves_ep += 1
                continue
    for scene in scenes:
        if scene == 'curve':
            curves += 1
    if curves <= 4:
        return 'straight'
    elif curves == _eps:
        return 'curve'
    else:
        return 'all'

def flat_lst(lsts):
    output = []
    for lst in lsts:
        output+= lst
    return output   

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def safemax(xs):
    return np.nan if len(xs) == 0 else np.max(xs)

class Model(object):
    def __init__(self, pi, reward_giver, nenv, stochastic=True):
        self.pi = pi
        self.nenv = nenv
        self.reward_giver = reward_giver
        self.stochastic = stochastic
    
    def step(self, obs):
        actions = []
        values = []
        rewards = []
        for i in range(self.nenv):
            ac, vpred = self.pi.act(self.stochastic, obs[i])
            reward = self.reward_giver.get_reward(obs[i], ac)
            actions.append(ac)
            values.append(vpred)
            rewards.append(reward[0][0])
        return actions, values, rewards

    def step_decision(self, obs):
        # here we assign a fake reward 0 to the output
        # and update reward in the belows
        actions = []
        values = []
        rewards = []
        for i in range(self.nenv):
            ac, vpred = self.pi.act(self.stochastic, obs[i])
            reward = 0
            actions.append(ac)
            values.append(vpred)
            rewards.append(reward)
        return actions, values, rewards

    def fake_rew(self, obs, ctrls):
        rewards = []
        for i in range(self.nenv):
            ctrls[i] = np.asarray(ctrls[i])
            reward = self.reward_giver.get_reward(obs[i], ctrls[i])
            # if terminal_list[i]:
            #     reward = [[-2]]
            rewards.append(reward[0][0])
        return rewards

# @ Junning: menv runner
class Runner(object):
    def __init__(self, env, model, nsteps, gamma, lam, length=800, rew_type=False, model_output='DM'):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.nenv = nenv
        self.obs = np.zeros((nenv, ) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self.obs[:] = env.reset()

        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.news = [[False] for _ in range(nenv)] # horizon & early reset
        self.terminals = [[False] for _ in range(nenv)] # early reset
        self._ep = int(nsteps/length)+1
        
        self.epinfo_buffer = deque([], maxlen=100)
        self.rew_type = rew_type
        self.model_output = model_output

    def run(self):
        mb_obs, mb_rewards, mb_true_rewards, mb_actions, mb_values, mb_news, mb_infos, mb_terminals = [], [], [], [], [], [], [], []
        mb_controls = []
        mb_current_pos = []
        mb_yaw = []
        # step for start
        epinfos = []
        start_action = [0, 1]

        if self.model_output == 'DM':
            _, v_init, _ = self.model.step_decision(self.obs)
        else:
            _, v_init, _ = self.model.step(self.obs)
        v_ep = [[] for i in range(self.nenv)]
        eprew_tmp = np.array([0. for i in range(self.nenv)])
        eprews_tmp = [[] for i in range(self.nenv)]

        for _step in range(self.nsteps):
            nb_actions = []
            # note that actions maybe decisions or controls
            # if actions are decisions, the controls can be
            # obtain from environment
            if self.model_output == 'DM':
                actions, values, rewards = self.model.step_decision(self.obs)
            else:
                actions, values, rewards = self.model.step(self.obs)

            mb_obs.append(self.obs.copy())
            mb_actions.append(np.asarray(actions, dtype=np.float32))
            mb_values.append(np.asarray(values, dtype=np.float32))
            mb_news.append(self.news)
            mb_terminals.append(self.terminals)
            # nb_actions = nb_actions.swapaxes(1, 0)
            self.obs[:], true_rewards, self.news, infos = self.env.step(actions)
            # ------------------------------------------------------------------
            self.env.venv.venv.envs[0].world.v_value = values
            controls = []
            current_pos_list = []
            yaw_list = []
            # terminal_list = []

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
                control = info.get('control')
                controls.append(control)
                current_pos_list.append(info.get('current_pos'))
                yaw_list.append(info.get('yaw'))
                # terminal_list.append(info.get('terminal'))
            mb_controls.append(np.asarray(controls, dtype=np.float32))
            mb_current_pos.append(np.asarray(current_pos_list, dtype=np.float32))
            mb_yaw.append(np.asarray(yaw_list, dtype=np.float32))
            # ------------------------------------------------------------------
            if self.model_output == 'DM':
                rewards = self.model.fake_rew(self.obs, controls)

            for i in range(len(self.env.venv.venv.envs)):
                if self.env.venv.venv.envs[i].terminal:
                    rewards[i] = -2
                    true_rewards[i] = -2
            
            self.env.venv.venv.envs[0].world.fake_reward = rewards
                # print(rewards, "fake REWARD")

            eprew_tmp += rewards
            self.terminals = np.array([info['terminal'] for info in infos])
            # add episode start id
            for __i in range(self.nenv):
                if self.news[__i]:
                    v_ep[__i].append(values[__i])
                    eprews_tmp[__i].append(eprew_tmp[__i])
                    eprew_tmp[__i] = 0.
            mb_true_rewards.append(true_rewards)
            mb_rewards.append(rewards)
            mb_infos.append(infos)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_news = np.asarray(mb_news, dtype=np.bool)
        mb_terminals = np.asarray(mb_terminals, dtype=np.bool)
        mb_true_rewards = np.asarray(mb_true_rewards)
        mb_infos = np.asarray(mb_infos)
        mb_controls = np.asarray(mb_controls)
        mb_current_pos = np.asarray(mb_current_pos)
        mb_yaw = np.asarray(mb_yaw)

        # last_values = []
        if self.model_output == 'DM':
            _, _values, _ = self.model.step_decision(self.obs)
        else:
            _, _values, _ = self.model.step(self.obs)
        last_values = _values
        last_values = np.asarray(last_values)
        # last_values = np.asarray(last_values, dtype=np.float32).swapaxes(1, 0)
        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonnew = 1.0 - self.news  # collision or maxlen horizon
                nextnonterminal = 1.0 - self.terminals  # collision
                nextvalues = last_values
            else:
                nextnonnew = 1.0 - mb_news[t + 1]
                nextnonterminal = 1.0 - mb_terminals[t + 1]
                nextvalues = mb_values[t + 1]
            if self.rew_type:
                delta = mb_true_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            else:
                delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonnew * lastgaelam
        mb_returns = mb_advs + mb_values

        # TODO: test sf01 function here, compare with stack
        # flatten axes 0 and 1, (nstep, nenv, d_shape) -> (nstep*nenv, d_shape)
        output = [*map(stack, (
        mb_obs, mb_current_pos, mb_yaw, mb_rewards, mb_returns, mb_news, mb_true_rewards, mb_actions, mb_values,
        mb_advs, mb_infos, mb_controls))]
        obs, current_pos, yaw, rews, returns, news, true_rews, acs, vpreds, advs = output[:-2]
        ctrls = output[-1]

        # episode statistics
        self.epinfo_buffer.extend(epinfos)

        mb_scenes = []
        for _i in range(self.nsteps):
            mb_scene_envs = []
            for _j in range(self.nenv):
                mb_scene_envs.append(mb_infos[_i][_j]['scene'])
            mb_scenes.append(mb_scene_envs)

        # flatten axes 0 and 1, (nstep, nenv, d_shape) -> (nstep*nenv, d_shape)
        v_ep, scenes, ep_rets = [*map(flat_lst, (v_ep, mb_scenes, eprews_tmp))]  # store v_ep after reset

        # log from epinfo: remember this is from 100 rolling buffer
        ep_true_rets = [epinfo['r'] for epinfo in self.epinfo_buffer]
        ep_lens = [epinfo['l'] for epinfo in self.epinfo_buffer]
        ep_v = safemean([epinfo['v'] for epinfo in self.epinfo_buffer])
        ep_acc = safemean([epinfo['acc'] for epinfo in self.epinfo_buffer])
        ep_left_offset = safemean([epinfo['left'] for epinfo in self.epinfo_buffer])
        ep_right_offset = safemean([epinfo['right'] for epinfo in self.epinfo_buffer])

        # record non-rolling, non-buffered info
        ep_true_lens = [ep['l'] for ep in epinfos]

        scene = detect_scene(scenes, ep_lens)
        return {"ob": obs, "rew": rews, "vpred": vpreds, "new": news, "truerew": true_rews, "v_ep": v_ep,
                "ac": acs, "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets, "scene": scene,
                "adv": advs, "tdlamret": returns, 'ep_v': ep_v, 'ep_acc': ep_acc, 'ep_left_offset': ep_left_offset,
                'ep_right_offset': ep_right_offset, "ep_true_lens": ep_true_lens, "ctrl": ctrls,
                'current_pos': current_pos, 'yaw': yaw}

def add_vtarg_and_adv(seg, gamma, lam, rew=False):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    if not rew:
        rew = seg["rew"]
    else:
        rew = seg["truerew"]

    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, reward_giver, expert_dataset, rank,
          pretrained, pretrained_weight, *,
          g_step, d_step, entcoeff, save_per_iter,
          ckpt_dir, log_dir, timesteps_per_batch, task_name,
          gamma, lam,
          max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, d_stepsize=3e-4, vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0,
          callback=None, load_model_path=None, search=False, search_mode='traj', scene='all',
          p_update=True, rew_type=False, train_mode='all', r_norm=False,still_std=False, model_output='DM'):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=(pretrained_weight != None))
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus

    if isinstance(env.action_space, spaces.MultiDiscrete):
        losses = [optimgain, meankl, entbonus, surrgain, meanent]
        loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]
    elif isinstance(env.action_space, spaces.Box):
        pi_mean = tf.reduce_mean(pi.pd.mean)
        pi_std = tf.reduce_mean(pi.pd.std)

        steer = tf.reduce_mean(pi.pd.mean[:, 0])
        steer_std = tf.reduce_mean(pi.pd.std[:, 0])
        if train_mode == "all":
            throttle_brake = tf.reduce_mean(pi.pd.mean[:, 1])
            throttle_brake_std = tf.reduce_mean(pi.pd.std[:, 1])
            losses = [optimgain, meankl, entbonus, surrgain, meanent, pi_mean, pi_std, steer, throttle_brake, steer_std, throttle_brake_std]
            loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy", "pi_mean", "pi_std", "steer", "throttle_brake", "steer_std", "throttle_brake_std"]
        elif train_mode == "steer":
            losses = [optimgain, meankl, entbonus, surrgain, meanent, pi_mean, pi_std, steer]
            loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy", "pi_mean", "pi_std", "steer"]
    
    dist = meankl
    
    all_var_list = pi.get_trainable_variables()
    if still_std:
        var_list = [v for v in all_var_list if v.name.startswith("pi/pol")]# or v.name.startswith("pi/logstd")]
    else:
        var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]

    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
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

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))
    
    # metadata file
    if expert_dataset is not None:
        metadata_file_loc = ckpt_dir+'/metadata.tsv'
        metadata_file = 'metadata.tsv'
        os.makedirs(os.path.dirname(metadata_file_loc), exist_ok=True)
        g_labels = [0 for i in range(g_step*timesteps_per_batch)]
        d_labels = expert_dataset.d_labels

        data_labels = g_labels+d_labels
        with open(metadata_file_loc, 'a') as f:
            for index, label in enumerate(data_labels):
                f.write("%s\n"%(label))
            
        # embedding settings
        OB_VIZ = 'Embedding_ob'
        ob_embedding = tf.get_variable(name=OB_VIZ, shape=[g_step*timesteps_per_batch+expert_dataset.obs.shape[0], ob_space.shape[0]])
        ob_ph = tf.placeholder(dtype=tf.float32, shape=[g_step*timesteps_per_batch+expert_dataset.obs.shape[0], ob_space.shape[0]], name="ob_ph") 
        assign_ob = tf.assign(ob_embedding, ob_ph)
        saver_em = tf.train.Saver({OB_VIZ: ob_embedding})
        config_ob = projector.ProjectorConfig()
        embedding_ob = config_ob.embeddings.add()
        embedding_ob.tensor_name = OB_VIZ 
        embedding_ob.metadata_path = metadata_file
        projector.visualize_embeddings(tf.summary.FileWriter(ckpt_dir), config_ob)

    ep_vs_global = deque(maxlen=40)
    ep_rets_global = deque(maxlen=40)
    ep_true_rets_global = deque(maxlen=40)

    @contextmanager
    def timed(msg):
        if rank == 0:
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
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    vfadam.sync()
    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    model = Model(pi, reward_giver, env.num_envs, stochastic=True)
    if expert_dataset is not None:
        seg_gen = Runner(env, model, timesteps_per_batch, gamma, lam, length=expert_dataset.num_length, rew_type=rew_type)
    else:
        seg_gen = Runner(env, model, timesteps_per_batch, gamma, lam, logger.num_length, rew_type=rew_type)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=100)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(reward_giver.loss_name)
    ep_stats = stats(["True_rewards", "Rewards", "Episode_length"])
    # if provide pretrained weight
    if pretrained_weight is not None:
        logger.log("Let's load the pretrained BC model.")
        logger.log("Some amazing things will happen.")
        U.load_variables(pretrained_weight, variables=pi.get_variables(), sess=tf.get_default_session())
    
    if load_model_path is not None:
        logger.log("Let's load the pretrained model")
        logger.log("For god sake, Who knows what will happen.")
        saver = tf.train.Saver(max_to_keep=5000)
        saver_best = tf.train.Saver()
        sess = tf.get_default_session()
        params = sess.run(pi.get_trainable_variables())
        saver.restore(tf.get_default_session(), load_model_path)
        params = sess.run(pi.get_trainable_variables())

    else:
        saver = tf.train.Saver(max_to_keep=5000)
        saver_best = tf.train.Saver()
    eptruerew_best = 0
    # if r_norm:
    #     rew_norm = RunningMeanStd()
    model_init = os.path.join(ckpt_dir, 'model_init')
    saver.save(tf.get_default_session(), model_init)

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        logger.log("********** Iteration %i ************" % iters_so_far)

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p
        # ------------------ Update G ------------------
        if p_update:
            logger.log("Optimizing Policy...")
            ob_g = []
            for _ in range(g_step):
                repeat = True
                while repeat:
                    with timed("sampling"):
                        seg = seg_gen.run()
                    if seg["scene"] == scene or scene == "all":
                        repeat = False
                logger.log("Scene :%s"%seg["scene"])
                
                # reward normalization
                # if r_norm:
                #     rew_norm.update(seg["rew"])
                #     seg["rew"] = (seg["rew"] - rew_norm.mean) / rew_norm.var

                # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
                ob_g.append(ob)
                ob_start = ob[0]
                vpredbefore = seg["vpred"]  # predicted value function before udpate
                atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

                if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

                args = seg["ob"], seg["ac"], atarg
                fvpargs = [arr[::5] for arr in args]

                assign_old_eq_new()  # set old parameter values to new parameter values
                with timed("computegrad"):
                    *lossbefore, g = compute_lossandgrad(*args)
                lossbefore = allmean(np.array(lossbefore))
                g = allmean(g)
                g_policy = g
                if np.allclose(g, 0):
                    logger.log("Got zero gradient. not updating")
                else:
                    with timed("cg"):
                        stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
                    assert np.isfinite(stepdir).all()
                    shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                    lm = np.sqrt(shs / max_kl)
                    # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                    fullstep = stepdir / lm
                    expectedimprove = g.dot(fullstep)
                    surrbefore = lossbefore[0]
                         
                    stepsize = 1.0
                    thbefore = get_flat()
                    for _ in range(10):
                        thnew = thbefore + fullstep * stepsize
                        set_from_flat(thnew)
                        meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
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
                        set_from_flat(thbefore)
                    if nworkers > 1 and iters_so_far % 20 == 0:
                        paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                        assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
                with timed("vf"):
                    for _ in range(vf_iters):
                        for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                                include_final_partial_batch=False, batch_size=128):
                            if hasattr(pi, "ob_rms"):
                                pi.ob_rms.update(mbob)  # update running mean/std for policy
                            g = allmean(compute_vflossandgrad(mbob, mbret))
                            vfadam.update(g, vf_stepsize)
                    g_vf = g
            g_losses = meanlosses
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
            ep_v, ep_acc, ep_left_offset, ep_right_offset = seg["ep_v"], seg["ep_acc"], seg["ep_left_offset"], seg["ep_right_offset"]

        else:
            seg = seg_gen.run()
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            ob_g = [ob]
        # ------------------ Update D ------------------
        if expert_dataset is not None:
            logger.log("Optimizing Discriminator...")
            logger.log(fmt_row(13, reward_giver.loss_name))
            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
            batch_size = len(ob) // d_step
            d_losses = []  # list of tuples, each of which gives the loss for a minibatch
            if model_output == 'DM':
                ac = seg["ctrl"]
            for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                        include_final_partial_batch=False,
                                                        batch_size=batch_size):
                if not p_update:
                    with timed("just update discriminator"):
                        ob_expert, ac_expert, search_prop = expert_dataset.obs, expert_dataset.acs, 0
                elif search:
                    with timed("searching batch"):
                        if search_mode == 'step':
                            ob_expert, ac_expert, search_prop = expert_dataset.search_batch_step(ob_batch, ac_batch)
                        elif search_mode == 'traj':
                            ob_expert, ac_expert, search_prop = expert_dataset.search_batch_traj(ob_start, batch_size, scene=scene)

                else:
                    ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch), scene=scene)
                # update running mean/std for reward_giver
                if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
                *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
                g_d = g
                d_adam.update(allmean(g), d_stepsize)
                d_losses.append(newlosses)

            # ------------------ Visualize Embedding ---------------
            if p_update:
                ob_g = np.array(ob_g)
                ob_g = np.reshape(ob_g, [-1, np.prod(ob_g.shape[2:])])
                ob_viz = np.concatenate([ob_g, expert_dataset.obs], axis=0) 
                sess = tf.get_default_session()
                sess.run(assign_ob, feed_dict={ob_ph: ob_viz})
                ob_name = 'model_ob_'+str(iters_so_far)+'.ckpt'
                saver_em.save(tf.get_default_session(), os.path.join(ckpt_dir, ob_name))

            logger.log(fmt_row(13, np.mean(d_losses, axis=0)))
            d_losses_name = reward_giver.loss_name
            d_losses_data = np.mean(d_losses, axis=0)
            kvs = [{name: data} for name, data in zip(d_losses_name, d_losses_data)]
            for kv in kvs:
                for k, v in kv.items():
                    logger.record_tabular(k, v)
        lrlocal = (seg["ep_true_lens"], seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        true_lens, lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        # Save model
        eptruerew_now = np.mean(true_rets)
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            modelname = 'model%d.ckpt'%iters_so_far
            fname = os.path.join(ckpt_dir, modelname)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver.save(tf.get_default_session(), fname)
        if rank == 0 and ckpt_dir is not None and eptruerew_now > eptruerew_best:
            modelname = 'modelbest.ckpt'
            fname = os.path.join(ckpt_dir, modelname)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver_best.save(tf.get_default_session(), fname)
            eptruerew_best = eptruerew_now
        eptruerew_last = eptruerew_now
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpLenMax", np.max(lenbuffer))
        if expert_dataset is not None:
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpTrueRewMax", np.max(true_rewbuffer))    
        logger.record_tabular("EpThisIter", len(true_lens))
        logger.record_tabular("EpVelocity", ep_v)
        logger.record_tabular("EpAcc", ep_acc)
        logger.record_tabular("EpLeftOffset", ep_left_offset)
        logger.record_tabular("EpRightOffset", ep_right_offset)

        corr_rew = np.corrcoef([seg["rew"], seg["truerew"]])[0][1]

        ep_rets = [ret for ret in seg["ep_rets"]]
        min_len = min(len(seg["v_ep"]), len(seg["ep_true_rets"]), len(ep_rets))
        for i in range(min_len):
            ep_vs_global.append(seg["v_ep"][i])
            ep_rets_global.append(ep_rets[i])
            ep_true_rets_global.append(seg["ep_true_rets"][i])
        corr_eprew = np.corrcoef([ep_vs_global, ep_rets_global])[0][1]
        corr_eptruerew = np.corrcoef([ep_vs_global, ep_true_rets_global])[0][1]
        logger.record_tabular("CorrRew", corr_rew)
        logger.record_tabular("CorrEpRew", corr_eprew)
        logger.record_tabular("CorrEpTrueRew", corr_eptruerew)

        episodes_so_far += len(true_lens)
        timesteps_so_far += sum(true_lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
