from . import VecEnvWrapper
from baselines.bench.monitor import ResultsWriter
import numpy as np
import time


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.tstart = time.time()
        self.results_writer = ResultsWriter(filename, header={'t_start': self.tstart})

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')

        # @Junning carla vehicle info
        # -----------------------------------------------
        self.epvs = [[] for i in range(self.num_envs)]
        self.epaccs = [[] for i in range(self.num_envs)]
        self.epleftoffset = [[] for i in range(self.num_envs)]
        self.eprightoffset = [[] for i in range(self.num_envs)]
        # -----------------------------------------------

        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        # @Junning: carla vehicle info
        # -----------------------------------------------        
        for i in range(len(infos)):
            self.epvs[i].append(infos[i]['v'])
            self.epaccs[i].append(infos[i]['acc'])
            self.epleftoffset[i].append(infos[i]['left_offset'])
            self.eprightoffset[i].append(infos[i]['right_offset'])
        # -----------------------------------------------

        newinfos = []
        # for (i, (done, ret, eplen, info)) in enumerate(zip(dones, self.eprets, self.eplens, infos)):
        for (i, (done, ret, eplen, epv, epacc, epleft, epright, info)) in enumerate(zip(dones, self.eprets, self.eplens, self.epvs, 
                                                                                        self.epaccs, self.epleftoffset, self.eprightoffset, 
                                                                                        infos)):
            info = info.copy()
            if done:
                # epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6), 'v': np.mean(epv),
                          'acc': np.mean(epacc), 'left': np.mean(epleft), 'right': np.mean(epright)}
                info['episode'] = epinfo
                self.eprets[i] = 0
                self.eplens[i] = 0

                # @Junning: carla vehicle info
                # -----------------------------------------------        
                self.epvs[i] = []
                self.epaccs[i] = []
                self.epleftoffset[i] = []
                self.eprightoffset[i] = []
                # @Junning: carla vehicle info
                # -----------------------------------------------        

                self.results_writer.write_row(epinfo)

            newinfos.append(info)

        return obs, rews, dones, newinfos
