from dataLoader import DataLoader
import numpy as np

class Environment:
    def __init__(self, popularity, cache_size, num_servers, num_files, reward_params, env_params, q_state):

        # Load requests
        # if isinstance(requests, DataLoader):
        #     self.requests = requests.get_requests()

        # if len(self.requests) <= cache_size:
        #     raise ValueError("The count of requests are too small. Try larger one.")

        self.popularity = popularity
        self.num_svrs = num_servers
        self.num_files = num_files
        self.reward_params = reward_params
        self.env_params = env_params
        self.q_state = q_state
        self.cache_size = cache_size

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
        d_old = self.calc_delay(ob)
        d_new = self.calc_delay(ob_new)

        if d_old >= d_new:
            return 1
        else:
            return 0


    def calc_delay(self, ob):
        # transmission delay + waiting delay
        total_dalay = 0.0
        wait_delay = 0.0
        matrix = ob['cache_state']

        for i in range(len(matrix)):
            T_i = 0
            wait_delay = ob['queue_state'][i] * self.env_params['r_iu']
            for f, x_if in enumerate(matrix[i]):
                p_if = ob['popularity'][i][f]
                f_size = self.get_size(f)
                t_iu = f_size / self.env_params['r_iu']

                if x_if == 1:
                    t_f = t_iu
                elif np.nonzero(matrix[:, f]):
                    h_cnt = 1   # graph 생성해서 hop number count 로 수정하기
                    t_ji = f_size / self.env_params['r_ij'] * h_cnt
                    t_f = t_iu + t_ji
                else:
                    t_ci = f_size / self.env_params['r_ci']
                    t_f = t_iu + t_ci
                T_i += p_if * t_f
            total_dalay += (T_i + wait_delay)

        return total_dalay


    def get_size(self, file):
        return self.env_params['file_size'][file]


    def get_observation(self):
        return dict(state=self.state,
                    cache_state=self.cache_mat.copy(),
                    queue_state=self.q_state.copy(),
                    popularity=self.popularity.copy()
                    )