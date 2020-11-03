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
from baselines.common.vec_env.vec_normalize import RewNorm

def traj_segment_generator(pi, env, reward_giver, horizon, stochastic, length=200, rew_normalizer=None, truerew_normalizer=None):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    terminal = True
    rew = 0.0
    true_rew = 0.0
    ob = env.reset()
    step = 0

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []
    ep_lens = []
    v_ep = []
    v_start_id = 0

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')

    # normalize reward 
    rews_norm = np.zeros(horizon, 'float32')
    truerews_norm = np.zeros(horizon, 'float32')

    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    terminals = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    scenes = deque([], maxlen=horizon)
    def detect_scene(scenes, ep_lens):
        _eps = len(ep_lens)
        curves_ep = 0
        _range = []
        start_idx = 0
        for _len in ep_lens:
            _range.append((start_idx, start_idx+_len))
            start_idx = start_idx + _len
        for start_idx, end_idx in _range:
            scenes = list(scenes)
            scene_ep = scenes[start_idx: end_idx]
            for scene in scene_ep:
                if scene == 'curve':
                    curves_ep += 1
                    continue
        if 'curve' not in scenes:
            return 'straight'
        elif curves_ep == _eps:
            return 'curve'
        else:
            return 'all'
    
    # @ Junning:
    # add physical reset

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            scene = detect_scene(scenes, ep_lens)
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news, "truerew": true_rews, "v_ep": v_ep,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new), "rew_norm": rews_norm, "truerew_norm": truerews_norm,
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets, "scene": scene, 'terminal': terminals}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            v_ep = []
            step = 0
            v_start_id = 0
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        terminals[i] = terminal
        acs[i] = ac
        prevacs[i] = prevac

        rew = reward_giver.get_reward(ob, ac)
        if rew_normalizer is not None:
            rew_norm = rew_normalizer.update(rew)

        ob, true_rew, new, info = env.step(ac)
        # @ Junning: 
        # new: collision or maxlen 
        # terminal: collision
        scene = info['scene']
        terminal = info['terminal']

        if truerew_normalizer is not None:
            truerew_norm = truerew_normalizer.update([true_rew])[0]

        scenes.append(scene)
        rews[i] = rew
        if rew_normalizer is not None:
            rews_norm[i] = rew_norm
        if truerew_normalizer is not None:
            truerews_norm[i] = truerew_norm
        true_rews[i] = true_rew
        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            v_ep.append(vpreds[v_start_id])
            v_start_id = step+1
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            time_ =time.time()
            ob = env.reset()
            time__ =time.time()
            if rew_normalizer is not None:
                rew_normalizer.reset()
            if truerew_normalizer is not None:
                truerew_normalizer.reset()
        t += 1
        step += 1

def add_vtarg_and_adv(seg, gamma, lam, rew=False, r_norm=True):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    terminatal = np.append(seg['terminal'], 0)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    if not rew:
        if r_norm:
            rew = seg["rew_norm"]
        else:
            rew = seg["rew"]
    else:
        if r_norm:
            rew = seg["truerew_norm"]
        else:
            rew = seg["truerew"]
    
    lastgaelam = 0
    for t in reversed(range(T)):
        nonnew = 1-new[t+1]
        nonterminal = 1-terminatal[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t] # collision
        gaelam[t] = lastgaelam = delta + gamma * lam * nonnew * lastgaelam # collision or maxlen
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
          p_update=True, rew_type=False, train_mode='all', r_norm=False
          ):

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
    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    assert len(var_list) == len(vf_var_list) + 1
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

    rew_normalizer = None
    truerew_normalizer = None
    if r_norm:
        rew_normalizer = RewNorm()
        truerew_normalizer = RewNorm()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, reward_giver, timesteps_per_batch, stochastic=True, length=expert_dataset.num_length, rew_normalizer=rew_normalizer, truerew_normalizer=truerew_normalizer)

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
        saver.restore(tf.get_default_session(), load_model_path)

    else:
        saver = tf.train.Saver(max_to_keep=5000)
        saver_best = tf.train.Saver()
    eptruerew_best = 0


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
                        seg = seg_gen.__next__()
                    if seg["scene"] == scene or scene == "all":
                        repeat = False
                logger.log("Scene :%s"%seg["scene"])

                add_vtarg_and_adv(seg, gamma, lam, rew_type, r_norm)
                # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
                ob_g.append(ob)
                ob_start = ob[0]
                vpredbefore = seg["vpred"]  # predicted value function before udpate
                logger.record_tabular('Atarg_mean', np.mean(atarg.mean()))
                logger.record_tabular('Atarg_std', np.mean(atarg.std()))
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
            logger.record_tabular('G_policy_norm', LA.norm(g_policy))
            logger.record_tabular('G_vf_norm', LA.norm(g_vf))
            g_losses = meanlosses
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        else:
            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            ob_g = [ob]
        
        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, reward_giver.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch

        # @ Junning: bug track, lack of transitions sample      
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
                logger.record_tabular('Search proption', search_prop)
            else:
                ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch), scene=scene)
            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            g_d = g
            d_adam.update(allmean(g), d_stepsize)
            d_losses.append(newlosses)
        logger.record_tabular('G_d_norm', LA.norm(g_d))  

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
        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        # Save model
        eptruerew_now = np.mean(true_rewbuffer)
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
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))

        corr_rew = np.corrcoef([seg["rew"], seg["truerew"]])[0][1]

        ep_rets = [ret[0][0] for ret in seg["ep_rets"]]
        for i in range(len(seg["v_ep"])):
            ep_vs_global.append(seg["v_ep"][i])
            ep_rets_global.append(ep_rets[i])
            ep_true_rets_global.append(seg["ep_true_rets"][i])
        corr_eprew = np.corrcoef([ep_vs_global, ep_rets_global])[0][1]
        corr_eptruerew = np.corrcoef([ep_vs_global, ep_true_rets_global])[0][1]
        logger.record_tabular("CorrRew", corr_rew)
        logger.record_tabular("CorrEpRew", corr_eprew)
        logger.record_tabular("CorrEpTrueRew", corr_eptruerew)
        logger.record_tabular('Vpred', np.mean(seg["vpred"]))
        logger.record_tabular('ADV', np.mean(seg["adv"]))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
