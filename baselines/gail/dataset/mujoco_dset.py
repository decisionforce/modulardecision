'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np
from copy import copy

class Dset(object):
    def __init__(self, inputs, labels, randomize, obs_abspos_yaw=None):
        self.inputs = inputs
        self.labels = labels
        self.obs_abspos_yaw = obs_abspos_yaw
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels       

class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True):
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
        self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])

        self.rets = traj_data['ep_rets'][:traj_limitation]
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, obs.shape[0])
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.randomize)
        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


CURRENT_ROAD_IDX = -7
CURRENT_LANE_IDX = -6

class Carla_Dset(object):

    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=200, randomize=True, num_trajectories=200, 
    num_length=200, data_shape=187, search=False, search_mode='traj', mode='all', feature='wp_car', train_mode="all", 
    dim="2d", resampling=0, episode_length=200):
        import pickle
        # if logger.scenario_name == "Merge_Env":
        expert_path = expert_path.split("-")[logger.select_scenario_id]
        # print("Merge ENV expert pathes: ", expert_path)
        loadfile = open(expert_path, 'rb')
        self.num_length = num_length
        self.traj_limitation = traj_limitation
        self.search_mode = search_mode
        obs = np.zeros(shape=(num_trajectories, num_length, data_shape))
        obs_abspos_yaw = np.zeros(shape=(num_trajectories, num_length, 4))
        if train_mode == "all":
            ac_len = 2
        elif train_mode == "steer":
            ac_len = 1
        else:
            raise NotImplementedError
        acs = np.zeros(shape=(num_trajectories, num_length, ac_len))
        print("Num of trajectories: ", num_trajectories)
        print("Num of length: ", num_length)
        wp_len, obj_len, road_len, ctrl_len = logger.wp_len, logger.obj_len, logger.road_len, logger.ctrl_len
        for i in range(num_trajectories):
            for j in range(num_length):
                sample_data = pickle.load(loadfile)
                if mode == 'all':
                    end_idx = -ctrl_len
                elif mode == 'wp':
                    end_idx = wp_len
                elif mode == 'wp_obj':
                    end_idx = obj_len + wp_len
                sample_ob = sample_data[:end_idx]
                sample_abspos_yaw = sample_data[end_idx:end_idx + 4]
                if train_mode == "all":
                    sample_ac = sample_data[-2:]
                elif train_mode == "steer":
                    sample_ac = sample_data[-2]
                obs[i][j] = sample_ob
                obs_abspos_yaw[i][j] = sample_abspos_yaw
                acs[i][j] = sample_ac

        self.resampling = resampling
        obs = np.array(obs)
        obs_abspos_yaw = np.array(obs_abspos_yaw)
        acs = np.array(acs)
        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        self.obs = np.reshape(obs, [-1, np.prod(obs.shape[2:])])
        self.obs_abspos_yaw = np.reshape(obs_abspos_yaw, [-1, np.prod(obs_abspos_yaw.shape[2:])])
        self.acs = np.reshape(acs, [-1, np.prod(acs.shape[2:])])

        if self.resampling != 0:
            self.obs = self.obs[::self.resampling]
            self.obs_abspos_yaw = self.obs_abspos_yaw[::self.resampling]
            self.acs = self.acs[::self.resampling]

        self.obs_all = copy(self.obs)
        self.acs_all = copy(self.acs)
        self.obs_abspos_yaw_all = copy(self.obs_abspos_yaw)
        
        if logger.scenario_name == "OverTake":
            self.keypoints = {"ST": [0], "TL": [10], "KL40": [30], "TR": [40], "KMidlane30": [66], "KMidlane25": [100]}
            #self.keypoints = {"ST": [0], "TL": [10], "KL40": [30], "TR": [68], "KMidlane30": [96], "KMidlane25": [100]}
            self.keystages = ["ST", "TL", "KL40", "TR", "KMidlane30", "KMidlane25"]
            if len(logger.checkkeys) == 1:
                check_frame_start = self.keypoints[logger.checkkeys[0]][0]
                check_frame_end = self.keypoints[logger.checkkeys[0]][0]+episode_length
            else:
                check_frame_start = self.keypoints[logger.checkkeys[0]][0]
                end_key = min(self.keystages.index(logger.checkkeys[-1])+1, len(self.keystages)-1)
                if self.keystages[end_key] != self.keystages[-1]:
                    check_frame_end = self.keypoints[self.keystages[end_key]][0]
                else:
                    check_frame_end = num_length
            obs_tmp = []        
            acs_tmp = []
            obs_abspos_yaw_tmp = []
            print("Check Frames: start, end", check_frame_start, check_frame_end)
            for i in range(len(self.acs)):
                if i % num_length >= check_frame_start and i % num_length < check_frame_end:
                    obs_tmp.append(self.obs[i])
                    acs_tmp.append(self.acs[i])
                    obs_abspos_yaw_tmp.append(self.obs_abspos_yaw[i])
            self.obs = np.array(obs_tmp)
            self.acs = np.array(acs_tmp)
            self.obs_abspos_yaw = np.array(obs_abspos_yaw_tmp) 
            print("Check Stage Observations: ", len(self.obs))
            print("Check Stage Actions: ", len(self.acs))

        self.d_labels = []
        for i in range(len(self.acs)):
            self.d_labels.append(1)

        if search_mode == 'traj':
            self.space = self.num_length
        else:
            self.space = 1
        self.idxs = [i for i in range(0, len(self.obs), self.space)]

        if search:
            self._map()
            self.search_space = copy(self.index)
        # if len(self.acs) > 2:
        #     self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs) == len(self.obs_abspos_yaw)
        self.num_traj = min(num_trajectories, self.obs.shape[0])
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(inputs=self.obs, labels=self.acs, randomize=self.randomize, obs_abspos_yaw=self.obs_abspos_yaw)
        # for behavior cloning
        self.train_set = Dset(inputs=self.obs[:int(self.num_transition * train_fraction), :],
                              labels=self.acs[:int(self.num_transition * train_fraction), :],
                              randomize=self.randomize,
                              obs_abspos_yaw=self.obs_abspos_yaw[:int(self.num_transition * train_fraction), :])
        self.val_set = Dset(inputs=self.obs[int(self.num_transition * train_fraction):, :],
                            labels=self.acs[int(self.num_transition * train_fraction):, :],
                            randomize=self.randomize,
                            obs_abspos_yaw=self.obs_abspos_yaw[int(self.num_transition * train_fraction):, :])
        self.log_info()

    def log_info(self):
        logger.log("Total trajectorues: %d" % self.num_traj)
        logger.log("Total transitions: %d" % self.num_transition)

    # def get_next_batch(self, batch_size, split=None):
    #     if split is None:
    #         return self.dset.get_next_batch(batch_size)
    #     elif split == 'train':
    #         return self.train_set.get_next_batch(batch_size)
    #     elif split == 'val':
    #         return self.val_set.get_next_batch(batch_size)
    #     else:
    #         raise NotImplementedError

    def get_next_batch(self, batch_size, split=None, scene="all"):
        if split is None:
            while True:
                ob_expert, ac_expert = self.dset.get_next_batch(batch_size)
                if scene == "straight" or scene == "curve":
                    scene_now = self._mode_traj(ac_expert)
                    if scene_now == scene:
                        return ob_expert, ac_expert
                elif scene == "all":
                    return ob_expert, ac_expert
                else:
                    return NotImplementedError
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError
    
    def _max_change(self, lst):
        change_x = (lst[2]-lst[0])**2 + (lst[4]-lst[2])**2 + (lst[6]-lst[4])**2
        change_y = (lst[3]-lst[1])**2 + (lst[5]-lst[3])**2 + (lst[7]-lst[5])**2
        if change_x > change_y:
            return lst[0]
        else:
            return lst[1]
    
    def _map(self):
        # road_id -> lane_id -> waypoint_start 
        index = [{} for i in range(400)]
        for i in range(0, len(self.obs), self.space):
            ob = self.obs[i]
            waypoint_start = self._max_change(ob)
            road_id = int(ob[CURRENT_ROAD_IDX])
            lane_id = int(ob[CURRENT_LANE_IDX])
            if lane_id not in index[road_id].keys():
                index[road_id].update({lane_id: [(i, waypoint_start)]})
            else:
                index[road_id][lane_id].append((i, waypoint_start))
        for traj in index:
            for key in traj.keys():
                traj[key] = sorted(traj[key], key=lambda x: x[1])
        self.index = index
        logger.log("Search space build")

    def _search_step(self, ob_g, print_diff=False, max_dis=160):
        def _diff(idx_e, wp_g):
            return ((idx_e[1] - wp_g) ** 2)**0.5
        def _sub(idx_e, wp_g):
            return idx_e[1]-wp_g
        wp_g = self._max_change(ob_g)
        road_id = int(ob_g[CURRENT_ROAD_IDX])
        lane_id = int(ob_g[CURRENT_LANE_IDX])
        search = 1
        if len(self.index[road_id]) == 0 or lane_id not in self.index[road_id].keys() or len(self.search_space[road_id][lane_id]) == 0:
            np.random.shuffle(self.idxs)
            min_id = self.idxs[0]
            search = 0
        else:
            search_space = self.search_space[road_id][lane_id]
            start_id = 0
            end_id = len(search_space)-1
            # binary search for nearest id
            while start_id <= end_id:
                mid = (start_id + end_id)//2
                if _diff(search_space[mid], wp_g) < max_dis:
                    break
                else:
                    if _sub(search_space[mid], wp_g) > 0:
                        end_id = mid - 1
                    else:
                        start_id = mid + 1
            min_id = search_space[mid][0]
            if self.search_mode == 'traj':
                self.search_space[road_id][lane_id].pop(mid)
        if print_diff:
            print("--------------- diff %f ---------------", _diff(search_space[min_id], wp_g))
        return min_id, search

    def search_batch_step(self, ob_batch, ac_batch, shuffle=True):
        ob_expert = np.zeros_like(ob_batch)
        ac_expert = np.zeros_like(ac_batch)
        search_times = 0
        for i in range(ob_batch.shape[0]):
            min_id, search = self._search_step(ob_batch[i])
            ob_expert[i], ac_expert[i] = self.obs[min_id], self.acs[min_id]
            search_times += search
        idxs = np.arange(len(ob_expert))
        if shuffle:
            np.random.shuffle(idxs) 
        return ob_expert[idxs, :], ac_expert[idxs, :], search_times/ob_batch.shape[0] 
    

    def _mode_traj(self, ac_e):
        steers = ac_e[:, 0]
        if np.mean(steers) < 0.01:
            return "straight"
        else:
            return "curve"

    def _mode_point(self, ac):
        steer = ac[0]
        if steer < 0.01:
            return "straight"
        else:
            return "curve"

    def search_batch_traj(self, ob_start, batch_size, shuffle=True, scene="all"):
        search_times = 0
        ob_expert = []
        ac_expert = []
        self.search_space = copy(self.index)
        for i in range(self.traj_limitation):
            search = True 
            while search:
                min_id, search = self._search_step(ob_start)
                ob_e, ac_e = self.obs[min_id: min_id+self.num_length], self.acs[min_id: min_id+self.num_length]
                if self._mode_traj(ac_e) == scene or scene=="all":
                    search = False
                    search_times += 1
            ob_expert.append(ob_e)
            ac_expert.append(ac_e)
        ob_expert = np.array(ob_expert)
        ac_expert = np.array(ac_expert)
        ob_expert = np.reshape(ob_expert, [-1, np.prod(ob_expert.shape[2:])])
        ac_expert = np.reshape(ac_expert, [-1, np.prod(ac_expert.shape[2:])])
        idxs = np.arange(batch_size)
        if shuffle:
            np.random.shuffle(idxs) 
        return ob_expert[idxs, :], ac_expert[idxs, :], search_times/self.traj_limitation

def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
