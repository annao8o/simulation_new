# from qLearning.dataLoader import DataLoader
from qLearning.config import *
import numpy as np


class Environment:
    def __init__(self, **kwargs):    # popularity, cache_size, num_servers, num_files, reward_params, data_lst, ctrl):

        # Load requests
        # if isinstance(requests, DataLoader):
        #     self.requests = requests.get_requests()

        # if len(self.requests) <= cache_size:
        #     raise ValueError("The count of requests are too small. Try larger one.")

        self.popularity = kwargs['popularity']
        self.data_lst = kwargs['data']
        self.num_svrs = kwargs['num_server']
        self.num_files = kwargs['num_data']
        self.reward_params = kwargs['reward']
        self.network_env = kwargs['network_env']
        #self.env_params = env_params
        self.cache_size = kwargs['cache_size']
        self.ctrl = kwargs['controller']

        # Counters
        self.total_cnt = 0
        self.hit_cnt = 0
        self.miss_cnt = 0

        # Cache
        self.cache_mat = np.zeros((self.num_svrs, self.num_files))

        # State & Action
        self.states = [(s, f) for s in range(self.num_svrs) for f in range(self.num_files)]
        self.n_states = len(self.states)
        self.state = None
        self.r_states = [(s, f) for s in range(self.num_svrs) for f in range(self.num_files)]

        self.actions = [0, 1]
        self.n_actions = len(self.actions)

        # print("num_states are", self.n_states, "\nnum_actions are", self.n_actions)


    # def add_servers(self, svr_list):
    #     self.svr_list = svr_list
    #     return self.svr_list


    # Display the current cache state
    def display(self):
        print("The cache matrix X is \n", self.cache_mat)
        print("The popularity is \n", self.popularity)


    def show_rate(self, w):
        if w == "miss":
            v = self.miss_cnt
        elif w == "hit":
            v = self.hit_cnt
        return v / self.total_cnt


    def hasDone(self):
        return np.any(np.count_nonzero(self.cache_mat) >= self.cache_size * self.num_svrs)


    def reset(self):
        # 환경 초기화
        self.total_cnt = 0
        self.miss_cnt = 0
        self.hit_cnt = 0
        self.cur_time = 0
        self.cache_mat = np.zeros((self.num_svrs, self.num_files))
        self.state = self.states[np.random.randint(0, self.n_states)]
        self.r_states = [(s, f) for s in range(self.num_svrs) for f in range(self.num_files)]

        return self.get_observation()


    def step(self, action):
        observation = self.get_observation()

        self.cache_mat[self.state] = action
        if action:
            for s in self.r_states:
                if s[1] == self.state[1]:
                    self.r_states.remove(s)

        self.check_capacity()

        if self.r_states:
            self.state = self.r_states[np.random.randint(0, len(self.r_states))]

        # Get observation
        observation_new = self.get_observation()

        reward = self.reward_func(observation, observation_new)

        return observation_new, reward, self.hasDone()


    def check_capacity(self):
        tmp = []
        if np.any(np.count_nonzero(self.cache_mat[self.state[0]]) >= self.cache_size):
            for s in self.r_states:
                if s[0] == self.state[0]:
                    tmp.append(s)
            for t in tmp:
                self.r_states.remove(t)


    def reward_func(self, ob, ob_new):
        d_old = self.calc_total_delay(ob)
        d_new = self.calc_total_delay(ob_new)
        # print(d_old, d_new)

        if d_old >= d_new:
            return 1
        else:
            return 0


    def get_hopCount(self, i, j_lst):
        rtt_lst = list()
        # print('J_lst:', j_lst)
        for j in j_lst:
            rtt_lst.append(self.ctrl.rtt_map[i][j])
        return min(rtt_lst)


    def calc_lambda_if(self, popularity, svr_idx, file_idx):
        # print('arrival rate: ', self.network_env['arrival rate'], 'cache mat:', self.cache_mat[svr_idx][file_idx])
        return self.network_env['arrival rate'] * self.cache_mat[svr_idx][file_idx] * popularity


    def calc_wait_time(self, ob):
        wait_time = np.zeros(self.num_svrs)
        for i in range(self.num_svrs):
            e1 = 0
            e2 = 0
            for f in range(self.num_files):
                lambda_if = self.calc_lambda_if(ob['popularity'][i][f], i, f)
                # print('popularity: ', ob['popularity'][i][f])
                # print('lambda: ', lambda_if)
                service_t = self.get_size(f) / self.network_env['bandwidth']
                e2 += pow(service_t, 2) * lambda_if
                e1 += service_t * lambda_if
            wait_time[i] = e2 / (2 * (1 - e1))
        return wait_time

    def calc_trans_time(self, ob):
        trans_time = np.zeros(self.num_svrs)
        matrix = ob['cache_state']

        for i in range(len(matrix)):
            T_i = 0.0
            # wait_delay = self.calc_wait_time(ob, i)
            for f, x_if in enumerate(matrix[i]):
                p_if = ob['popularity'][i][f]
                f_size = self.get_size(f)
                t_iu = f_size / env_params['r_iu']
                #print(f_size, env_params['r_iu'])
                if x_if == 1:
                    t_f = t_iu
                elif list(np.nonzero(matrix[:, f])[0]):
                    # print(np.nonzero(matrix[:, f]))
                    j_lst = list(np.nonzero(matrix[:, f])[0])   #f를 가지고 있는 neighbor server list
                    h_cnt = self.get_hopCount(i, j_lst)
                    t_ji = f_size / env_params['r_ij'] * h_cnt
                    t_f = t_iu + t_ji
                else:
                    t_ci = f_size / env_params['r_ci']
                    t_f = t_iu + t_ci
                T_i += p_if * t_f
            #print(T_i)
            trans_time[i] = T_i
        return trans_time

    def calc_total_delay(self, ob):
        # transmission delay + waiting delay + service time
        trans_time = self.calc_trans_time(ob)
        wait_time = self.calc_wait_time(ob)
        # service_time = np.sum([self.get_size(f) / self.network_env['bandwidth'] for f in range(self.num_files)]

        # print(trans_time)
        # print(wait_time)
        # print(service_time)

        sum_delay = trans_time + wait_time # + service_time
        total_dalay = np.sum(sum_delay)
        # print(total_dalay)

        return total_dalay


    def get_size(self, file):
        #return self.env_params['file_size'][file]
        # print(self.ctrl.d_size_map)
        return self.ctrl.d_size_map[file]


    def get_observation(self):
        return dict(state=self.state,
                    cache_state=self.cache_mat.copy(),
                    popularity=self.popularity.copy()
                    )