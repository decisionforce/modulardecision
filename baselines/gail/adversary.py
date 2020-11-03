'''
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
'''
import tensorflow as tf
import numpy as np

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U
import pickle
import time
from baselines import logger
import math

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

class TransitionClassifier(object):
    def __init__(self, env, hidden_size, entcoeff=0.001, lr_rate=1e-3, scope="adversary", adversary_hidden_layer=1, sn=True, d_actv="relu", model="origin", 
                 detect=False, train_mode="all", dim="2d", model_output="DM"):
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.actions_shape = env.action_space.shape
        self.input_shape = tuple([o+a for o, a in zip(self.observation_shape, self.actions_shape)])
        self.num_actions = env.action_space.shape[0]
        self.hidden_size = hidden_size
        self.train_mode = train_mode
        self.model_output = model_output
        self.build_ph()
        # Build grpah

        if model == "origin":
            self.gl=generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False, num_layers=adversary_hidden_layer, sn=sn, d_actv=d_actv)
            self.el=expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True, num_layers=adversary_hidden_layer, sn=sn, d_actv=d_actv)
        elif model == "separate":
            self.gl=generator_logits = self.build_graph_separate(self.generator_obs_ph, self.generator_acs_ph, reuse=False, num_layers=adversary_hidden_layer, sn=sn, d_actv=d_actv)
            self.el=expert_logits = self.build_graph_separate(self.expert_obs_ph, self.expert_acs_ph, reuse=True, num_layers=adversary_hidden_layer, sn=sn, d_actv=d_actv)            
        else:
            raise NotImplementedError
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff*entropy
        
        # Define saliency for different parts of the observation
        wp_len = logger.wp_len
        obj_len = logger.obj_len
        #saliency_ob_idx = wp_len
        #saliency_va_idx = [wp_len, wp_len+obj_len]
        #saliency_wp = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, :saliency_ob_idx]))
        #saliency_va = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, saliency_va_idx[0]:saliency_va_idx[1]]))
        saliency_ob_idx = wp_len
        saliency_obj_idx = [wp_len, wp_len+obj_len]
        saliency_egoa_idx = logger.egoa_idx
        saliency_egov_idx = logger.egov_idx
        saliency_zombiea_idx = logger.zombiea_idx
        saliency_zombiev_idx = logger.zombiev_idx
        saliency_zombiebx_idx = logger.zombiebx_idx

        saliency_wp = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, :saliency_ob_idx]))
        saliency_obj = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, saliency_obj_idx[0]:saliency_obj_idx[1]]))
        saliency_egoav = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, saliency_egoa_idx[0]:saliency_egov_idx[1]]))
        saliency_egoa = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, saliency_egoa_idx[0]:saliency_egoa_idx[1]]))
        saliency_egov = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, saliency_egov_idx[0]:saliency_egov_idx[1]]))
        saliency_zombieav = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, saliency_zombiea_idx[0]:saliency_zombiev_idx[1]]))
        saliency_zombiea = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, saliency_zombiea_idx[0]:saliency_zombiea_idx[1]]))
        saliency_zombiev = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, saliency_zombiev_idx[0]:saliency_zombiev_idx[1]]))
        saliency_zombiebx = tf.reduce_mean(tf.abs(self.saliency_ob[0][:, saliency_zombiebx_idx[0]:saliency_zombiebx_idx[1]]))
        self.saliencies = [saliency_wp, saliency_egoa, saliency_egov, saliency_zombiea, saliency_zombiev, saliency_zombiebx]
        if model == "origin":
            if train_mode == "all":
                saliency_throttle_brake = tf.reduce_mean(tf.abs(self.saliency_ac[0][:, 1]))
                saliency_steer = tf.reduce_mean(tf.abs(self.saliency_ac[0][:, 0]))
                self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc, saliency_wp, saliency_obj, saliency_steer, saliency_throttle_brake, 
                               saliency_egoav, saliency_egoa, saliency_egov, saliency_zombieav, saliency_zombiea, saliency_zombiev, saliency_zombiebx]
                self.loss_name = ["generator_loss", "expert_loss", "entropy_d", "entropy_loss", "generator_acc", "expert_acc", "saliency_wp", "saliency_obj", "saliency_steer", "saliency_throttle_brake",
                               "saliency_egoav", "saliency_egoa", "saliency_egov", "saliency_zombieav", "saliency_zombiea", "saliency_zombiev", "saliency_zombiebx"]
            elif train_mode == "steer":
                saliency_steer = tf.reduce_mean(tf.abs(self.saliency_ac[0]))
                self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc, saliency_wp, saliency_obj, saliency_steer, 
                               saliency_egoav, saliency_egoa, saliency_egov, saliency_zombieav, saliency_zombiea, saliency_zombiev, saliency_zombiebx]
                self.loss_name = ["generator_loss", "expert_loss", "entropy_d", "entropy_loss", "generator_acc", "expert_acc", "saliency_wp", "saliency_obj", "saliency_steer", 
                                "saliency_egoav", "saliency_egoa", "saliency_egov", "saliency_zombieav", "saliency_zombiea", "saliency_zombiev", "saliency_zombiebx"]                
        elif model == "separate":
            saliency_ob_mid = tf.reduce_mean(tf.abs(self.saliency_ob_mid))
            if train_mode == "all":
                saliency_steer = tf.reduce_mean(tf.abs(self.saliency_ac[:, 0]))
                saliency_throttle_brake = tf.reduce_mean(tf.abs(self.saliency_ac[:, 1]))  
                self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc, saliency_wp, saliency_obj, saliency_ob_mid, saliency_steer, saliency_throttle_brake]
                self.loss_name = ["generator_loss", "expert_loss", "entropy_d", "entropy_loss", "generator_acc", "expert_acc", "saliency_wp", "saliency_obj", "saliency_ob_mid", "saliency_steer", "saliency_throttle_brake"]
            elif train_mode == "steer":    
                saliency_steer = tf.reduce_mean(tf.abs(self.saliency_ac))
                self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc, saliency_wp, saliency_obj, saliency_ob_mid, saliency_steer]
                self.loss_name = ["generator_loss", "expert_loss", "entropy_d", "entropy_loss", "generator_acc", "expert_acc", "saliency_wp", "saliency_obj", "saliency_ob_mid", "saliency_steer"]
        
        # Loss + Accuracy terms
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
        var_list = self.get_trainable_variables()
        
        self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
                                    self.losses + [U.flatgrad(self.total_loss, var_list)])
        if detect:
            dir_name = logger.ckpt_dir_name
            self.outfile = open(dir_name+'/'+'test.pkl','wb')

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")
        if self.model_output == 'DM':
            self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + (2, ), name="actions_ph")
            self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + (2, ), name="expert_actions_ph")
        else:
            self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")
            self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert_actions_ph")

    def build_graph(self, obs_ph, acs_ph, reuse=False, num_layers=1, sn=True, d_actv="relu"):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean / self.obs_rms.std)
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition

            normalizer_fn = None
            if sn:
                normalizer_fn = spectral_norm
            if d_actv == "tanh":
                activation_fn = tf.nn.tanh
            elif d_actv == "relu":
                activation_fn = tf.nn.relu
            elif d_actv == "lrelu":
                activation_fn = tf.nn.leaky_relu
            else:
                print("Not available activation type")
                raise Exception
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            p_h = p_h1
            for i in range(num_layers):
                p_h = tf.contrib.layers.fully_connected(p_h, self.hidden_size, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            logits = tf.contrib.layers.fully_connected(p_h, 1, activation_fn=tf.identity, normalizer_fn=normalizer_fn)
            self.saliency_ob = tf.gradients(logits, obs_ph)
            self.saliency_ac = tf.gradients(logits, acs_ph)
        return logits

    def build_graph_separate(self, obs_ph, acs_ph, reuse=False, num_layers=1, sn=True, d_actv="relu"):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean / self.obs_rms.std)
            obs = obs[:, :63]
            va = obs[:, 64:64+8]

            _input = obs
            normalizer_fn = None
            if sn:
                normalizer_fn = spectral_norm
            if d_actv == "tanh":
                activation_fn = tf.nn.tanh
            elif d_actv == "relu":
                activation_fn = tf.nn.relu
            elif d_actv == "lrelu":
                activation_fn = tf.nn.leaky_relu
            else:
                print("Not available activation type")
                raise Exception
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            p_h = p_h1
            for i in range(num_layers-1):
                p_h = tf.contrib.layers.fully_connected(p_h, self.hidden_size, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            p_h = tf.contrib.layers.fully_connected(p_h, 2, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
            p_h = tf.concat([p_h, va, acs_ph], axis=1)
            logits = tf.contrib.layers.fully_connected(p_h, 1, activation_fn=tf.identity, normalizer_fn=normalizer_fn)
            self.saliency_ob = tf.gradients(logits, obs_ph)
            self.saliency_ob_mid = tf.gradients(logits, p_h)[0][:, :-2]
            if self.train_mode == "all":
                self.saliency_ac = tf.gradients(logits, p_h)[0][:, -2:]
            elif self.train_mode == "steer":
                self.saliency_ac = tf.gradients(logits, p_h)[0][:, -1]
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward

    def detect_error(self, obs, acs, type_ob="g"):
        def sigmoid(x):
            return 1./(1. + math.exp(-x))
        sess = tf.get_default_session()
        error = 0.
        if type_ob == "g":
            feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
            logits = sess.run(self.gl, feed_dict)
        elif type_ob == "e":
            feed_dict = {self.expert_obs_ph: obs, self.expert_acs_ph: acs}
            logits = sess.run(self.el, feed_dict)
        tmp = []
        for i in range(len(logits)):
            p = sigmoid(logits[0])
            tmp.append(p)
        logits = tmp
        for i in range(len(logits)):
            if (logits[i] > 0.5 and type_ob == "g") or (logits[i] < 0.5 and type_ob == "e"):
                pickle.dump([obs[i], acs[i], type_ob], self.outfile)
                error += 1.
        total_number = len(obs)
        return error, total_number

    def get_saliency(self, obs_exp, acs_exp, obs_ctfal, acs_ctfal):
        sess = tf.get_default_session()
        feed_dict = {self.generator_obs_ph: obs_ctfal, self.generator_acs_ph: acs_ctfal, self.expert_obs_ph: obs_exp, self.expert_acs_ph: acs_exp}
        saliencies_values = sess.run(self.saliencies, feed_dict)
        saliencies_names = ["saliency_evaluate_wp", "saliency_evaluate_egoa", "saliency_evaluate_egov", "saliency_evaluate_zombiea", "saliency_evaluate_zombiev", "saliency_evaluate_zombiebx"]
        saliencies = {saliencies_names[i]: saliencies_values[i] for i in range(len(saliencies_values))}
        return saliencies

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w_shape[0] = -1
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm
