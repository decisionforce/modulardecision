import numpy as np
from multiprocessing import Process, Pipe
from . import VecEnv, CloudpickleWrapper
import multiprocessing

def worker(remote, parent_remote, env_fn_wrapper, host, port, args):
    parent_remote.close()
    env = env_fn_wrapper.x(host, port, args)
    env.wait_for_reset = False
    # try:
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError
    # except KeyboardInterrupt:
    #     print('SubprocVecEnv worker: got KeyboardInterrupt')
    # finally:
    #     env.close()

def worker_pool(remote, parent_remote, env_fn_wrapper, host, port, args):
    parent_remote.close()
    env = env_fn_wrapper.x(host, port, args)
    env.wait_for_reset = False
    # try:
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            # if done:
            #     ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'reset_ob':
            env.wait_for_reset = True
        elif cmd == 'reset_control':
            ob = env.reset()
            env.reset_ob = ob
            env.wait_for_reset = False
        elif cmd == 'wait_for_reset':
            remote.send((env.wait_for_reset))
        elif cmd == 'get_reset_ob':
            remote.send((env.reset_ob))
        else:
            raise NotImplementedError

class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, hosts=None, ports=None, argss=None):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn), host, port, args))
                   for (work_remote, remote, env_fn, host, port, args) in zip(self.work_remotes, self.remotes, env_fns, hosts, ports, argss)]
        

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

class SubprocVecEnvPool(VecEnv):
    
    def __init__(self, env_fns, spaces=None, hosts=None, ports=None, argss=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker_pool, args=(work_remote, remote, CloudpickleWrapper(env_fn), host, port, args))
                   for (work_remote, remote, env_fn, host, port, args) in zip(self.work_remotes, self.remotes, env_fns, hosts, ports, argss)]
        
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.ts = np.zeros(len(env_fns), dtype='int')
        self.actions = None
        self.env_now = 0
        self.reset_obs = [None, None, None]

    def step_async(self, actions):
        self.actions = actions
        self._assert_not_closed()
        e = self.env_now
        action = actions[e]
        remote = self.remotes[e]
        remote.send(('step', action))
        self.waiting = True

    # async update step 
    def step_wait(self):
        # @TODO: env pool, revise the runner structure or change the step counts
        e = self.env_now
        remote = self.remotes[e]
        result = remote.recv()
        ob, rew, done, info = result
        self.ts[e] += 1
        if done:
            remote = self.remotes[e]
            remote.send(('reset_ob', done))
            remote.send(('reset_control', done))
            self.ts[e] = 0
        self.actions = None
        return np.array([ob]), np.array([rew]), np.array([done]), [info]

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.array([remote.recv() for remote in self.remotes])

    def is_reset(self, e):
        remote = self.remotes[e]
        remote.send(('wait_for_reset', None))
        wait_for_reset = remote.recv()
        return wait_for_reset

    def get_reset_ob(self, e):
        remote = self.remotes[e]
        remote.send(('get_reset_ob', None))
        ob = remote.recv()
        return ob

    def close(self):
        return

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
