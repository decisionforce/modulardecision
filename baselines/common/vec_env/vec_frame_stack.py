from . import VecEnvWrapper
import numpy as np
from gym import spaces


class VecFrameStack(VecEnvWrapper):
    """
    Vectorized environment base class
    """
    #def __init__(self, venv, nstack):
    def __init__(self, venv, nstack):
        #
        self.venv = venv
        self.nstack = nstack
        # @Lanxin Lei
        # our env must use [0], or will meet wrong in baselines
        # venv.observation_space = venv.observation_space[0]
        wos = venv.observation_space # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros(((venv.num_envs, )+low.shape), low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        #import pdb; pdb.set_trace()
        for (i, new) in enumerate(news):
            ### 
            if new:
                self.stackedobs[i] = 0
        #@Lanxin Lei
        #ValueError: could not broadcast input array from shape (16,1,77) into shape (16,77)
        #self.stackedobs[..., -obs.shape[-1]:] = obs
        #import pdb; pdb.set_trace()
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        #@Lanxin Lei
        #ValueError: could not broadcast input array from shape (16,1,77) into shape (16,77)
        #self.stackedobs[..., -obs.shape[-1]:] = obs
        #
        #for index in range(2):
        #import pdb; pdb.set_trace()
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def close(self):
        self.venv.close()

    #add a render()
    #@Lanxin Lei
    def render(self):
        self.venv.render()

# class VecFrameStack(VecEnvWrapper):
#     """
#     Vectorized environment base class
#     """
#     #def __init__(self, venv, nstack):
#     def __init__(self, venv, nstack, num_agent):
#         self.num_agent = num_agent
#         #
#         self.venv = venv
#         self.nstack = nstack
#         # @Lanxin Lei
#         # our env must use [0], or will meet wrong in baselines
#         # venv.observation_space = venv.observation_space[0]
#         wos = venv.observation_space # wrapped ob space
#         low = np.repeat(wos.low, self.nstack, axis=-1)
#         high = np.repeat(wos.high, self.nstack, axis=-1)
#         self.stackedobs = np.zeros(((venv.num_envs,self.num_agent,)+low.shape), low.dtype)
#         observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
#         VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
#         #import pdb; pdb.set_trace()
#         for (i, new) in enumerate(news):
#             ### 
#             for j,done in enumerate(new):
#                 #import pdb; pdb.set_trace()
#                 if done:
#                     self.stackedobs[i,j] = 0
#         #@Lanxin Lei
#         #ValueError: could not broadcast input array from shape (16,1,77) into shape (16,77)
#         #self.stackedobs[..., -obs.shape[-1]:] = obs
#         #import pdb; pdb.set_trace()
#         self.stackedobs[..., -obs.shape[-1]:] = obs
#         return self.stackedobs, rews, news, infos

#     def reset(self):
#         """
#         Reset all environments
#         """
#         obs = self.venv.reset()
#         self.stackedobs[...] = 0
#         #@Lanxin Lei
#         #ValueError: could not broadcast input array from shape (16,1,77) into shape (16,77)
#         #self.stackedobs[..., -obs.shape[-1]:] = obs
#         #
#         #for index in range(2):
#         #import pdb; pdb.set_trace()
#         self.stackedobs[..., -obs.shape[-1]:] = obs
#         return self.stackedobs

#     def close(self):
#         self.venv.close()

#     #add a render()
#     #@Lanxin Lei
#     def render(self):
#         self.venv.render()
