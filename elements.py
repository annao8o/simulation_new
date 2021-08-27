import numpy as np
from numpy.linalg import norm
import networkx as nx
from itertools import combinations
import random
from datetime import timedelta
from collections import deque
from configure import *
import matplotlib.pyplot as plt


'''
def default_select_func(**kwargs):
    cn = kwargs['cooperative network']
    data = kwargs['data']

    candidates = cn.svr_lst
    candidates = np.logical_and(candidates, np.logical_not(cn.ctrl.caching_map[:, data.id]))
    candidates = cn.ctrl.usable_storage * candidates
    candidates = candidates > data.size
    if np.any(candidates > 0):
        min_arr = np.min(cn.arrival_rate_map[candidates])   #arrival rate가 작은 server들
        min_candidates = np.where(np.logical_and(cn.arrival_rate_map == min_arr, candidates))
        return np.random.choice(min_candidates[0])
    else:
        return -1

def select_func_using_y():
    return
'''

def cos_sim(A, B):
    return np.dot(A, B) / (norm(A)*norm(B))


class Data:
    def __init__(self, id, size):
        self.id = id
        self.size = size

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return self.idx == other.idx
        else:
            return False

class RequestedData(Data):
    def __init__(self, data, request_time, near_svr=None):
        super().__init__(data.id, data.size)
        self.request_time = request_time
        self.near_svr = near_svr
        self.queueing_time = None

    def set_queueing(self, T):
        self.queueing_time = T

    def __eq__(self, other):
        if type(self) == type(other):
            return super().__eq__(other) and self.request_time == other.request_time
        else:
            return False


class User:
    def __init__(self, id):
        self.id = id
        self.pref_vec = None    #(pdf, cdf)
        self.user_type = None
        self.activity = None
        self.location = None

    def set_char(self, user_type, pdf, cdf, activity, location):
        self.pref_vec = (pdf, cdf)  #pdf.ravel()
        self.user_type = user_type
        self.activity = activity
        self.location = location

    def get_pref(self):
        return self.pref_vec[0]

    def print_user_info(self):
        print("user {} >>> user type {}\n{}".format(self.id, self.user_type, self.pref_vec[0]))
    # def request(self):
    #     f = np.random.random()
    #     content = np.searchsorted(self.pref_vec[1], f) - 1
    #
    #     hit_lst = self.region.request(self.user_type, content)
    #
    #     return content, hit_lst

    @staticmethod
    def calc_rtt(destination):
        if destination == User:
            return timedelta(seconds=0)
        else:
            return destination.calc_rtt(destination)


class MECServer:
    def __init__(self, id, controller):
        self.id = id
        self.ctrl = controller
        self.location = None
        self.capacity = cache_capacity
        self.storage_usage = 0
        self.queue = deque()
        self.popularity = None
        self.service_time = 0
        self.cn = None

    def set_popularity(self, p):
        self.popularity = p

    def calc_storage_usage(self):
        self.storage_usage = np.sum(self.ctrl.d_size_map * self.ctrl.caching_map[self.id, :])
        return self.storage_usage

    def get_usable_storage(self):
        return self.capacity - self.storage_usage

    def data_store(self, data):
        if data.size < self.get_usable_storage():
            self.ctrl.caching_map[self.id, data.id] = 1
            self.calc_storage_usage()
            self.ctrl.usable_storage[self.id] = self.get_usable_storage()
            return 1
        else:
            return 0

    def set_CN(self, cn):
        self.cn = cn

    def check_CN(self, data):    #check whether the data is stored in the sub-region
        return self.cn.isContain(data)

    def pop_request(self):
        if self.queue:
            return self.queue.popleft()
        else:
            return -1

    def processing_end(self, time=None, **_):
        self.processing_event = None
        kwargs = self.pop_request()
        if kwargs != -1:
            self.processing_event = kwargs['data']
            kwargs['simulator'].make_process_event(self, **kwargs)

    def calc_rtt(self, destination=None):
        if destination is None or destination is User:
            return timedelta(seconds=self.rtt)
        else:
            return timedelta(seconds=self.ctrl.rtt_map[self.id, destination.id])

    def calc_service_time(self):
        c_map = self.ctrl.caching_map[self.id, :]
        self.service_time = 0

        for i in range(len(c_map)):
            T_i = 0
            for f, x_if in enumerate(c_map[i]):
                p_if = self.popularity[f]
                f_size = self.ctrl.d_size_map[f]
                t_iu = f_size / env_params['r_iu']

                if x_if == 1:
                    t_f = t_iu
                elif np.nonzero(c_map[:, f]):
                    h_cnt = 1   # graph 생성해서 hop number count 로 수정하기
                    t_ji = f_size / self.env_params['r_ij'] * h_cnt
                    t_f = t_iu + t_ji
                else:
                    t_ci = f_size / self.env_params['r_ci']
                    t_f = t_iu + t_ci
                T_i += p_if * t_f
            total_dalay += (T_i + wait_delay)

        return

    def calc_wait_time(self):

        return


    def get_sample(self, size=None):
        zeta = np.r_[0.0, np.cumsum(self.popularity)]
        cdf = [x / zeta[-1] for x in zeta]

        if size is None:
            f = random.random()
        else:
            f = np.random.random(size)

        return np.searchsorted(cdf, f) - 1

    def append_request(self, **kwargs):
        kwargs['data'].set_queueing(kwargs['simulator'].T)
        self.queue.append(kwargs)
        if self.processing_event is None:
            self.processing_end()
        if len(self.queue) == 1:
            return 1
        else:
            return 0

    def clear(self):
        self.processing_event = None
        self.queue.clear()


class Controller:   # controller
    def __init__(self):
        self.num_svr = None
        self.num_data = None
        self.svr_lst = list()
        self.data_lst = list()
        self.CN_lst = list()
        self.caching_map = None
        self.graph = None
        self.d_size_map = None
        self.usable_storage = None


    def set_env(self, num_svr, num_data, request_info):
        self.num_svr = num_svr
        self.num_data = num_data
        self.caching_map = np.zeros((num_svr, num_data), dtype=np.bool_)
        self.d_size_map = np.zeros(num_data, dtype=np.int_)
        self.requests = request_info
        self.usable_storage = np.zeros(self.num_svr)

    def create_env(self, data_lst):
        g = nx.Graph()

        if type(data_lst) == list:
            self.data_lst = data_lst
        elif type(data_lst) == dict:
            self.data_lst = list(data_lst.values())
        else:
            raise Exception("wrong type error: %s is not acceptable for data list" % type(data_lst))

        for i in range(self.num_svr):
            svr = MECServer(i, self)
            self.svr_lst.append(svr)
            g.add_node(i)
            self.usable_storage[i] = svr.get_usable_storage()

        p = 0.5
        for u, v in combinations(g, 2):
            if random.random() < p:
                g.add_edge(u, v, rtt=random.uniform(0,1)*0.001+0.001)
        self.graph = g
        nx.draw(self.graph)
        plt.show()


    def set_popularity(self, popularity_lst):
        for i in range(len(popularity_lst)):
            self.svr_lst[i].set_popularity(popularity_lst[i, :])

    def set_svr_cache(self, svr_idx, cache_item):
        if type(cache_item) == np.ndarray:
            self.caching_map[svr_idx, :] = cache_item
        elif type(cache_item) == list:
            if len(cache_item) == self.env['num data']:
                self.caching_map[svr_idx, :] = np.array(cache_item)
            else:
                raise Exception("array dimension unmatch: cache data must interger or array")

        elif type(cache_item) == int:
            self.caching_map[svr_idx, cache_item] = 1
        else:
            raise Exception("wrong type error: cache data must interger or array")

    def check_cache_in_svr(self, svr_idx, data_idx):
        return self.caching_map[svr_idx, data_idx]

    def check_cache_in_CN(self, CN_matrix, data_idx):
        return self.caching_map[:, data_idx]@CN_matrix

    def make_CN(self, theta):
        checkList = [i for i in range(self.num_svr)]
        while checkList:
            popularity_map = list()
            cn = CooperativeNet(self.num_svr, self)
            i = random.choice(checkList)
            checkList.remove(i)
            cn.add_svr(i)
            self.svr_lst[i].set_CN(cn)
            popularity_map.append(self.svr_lst[i].popularity)

            # print(i, list(self.graph.neighbors(i)))
            # print(self.graph.edges())
            for n in list(self.graph.neighbors(i)):
                if n in checkList:
                    if self.svr_lst[n].cn is None:
                        sim = cos_sim(self.svr_lst[i].popularity, self.svr_lst[n].popularity) # calculate the cosine similarity between popularity of servers
                        if sim >= theta:
                            checkList.remove(n)
                            cn.add_svr(n)
                            self.svr_lst[n].set_CN(cn)
                            popularity_map.append(self.svr_lst[n].popularity)
                        else:
                            checkList.remove(n)
            self.CN_lst.append(cn)



    def rtt_mapping(self):
        self.rtt_map = np.zeros((self.num_svr, self.num_svr), dtype=np.float_)
        for i in range(self.num_svr):
            for j in range(i+1, self.num_svr):
                length = nx.shortest_path_length(self.graph, i, j, weight='rtt')
                self.rtt_map[i, j] = length
                self.rtt_map[j, i] = length

    def init_caching(self):
        self.caching_map = np.zeros_like(self.caching_map) if any(self.caching_map > 0) else self.caching_map

        for svr in self.svr_lst:
            svr.storage_usage = 0
            self.usable_storage[svr.id] = svr.get_usable_storage()
            # svr.lambda_i = 0.0
            svr.clear()

        for cn in self.CN_lst:
            result = cn.caching_policy()
            for r in range(len(result)):
                self.set_svr_cache(cn.svr_lst[r].id, result[r])

        return self.caching_map

    def clearCN(self):
        self.CN_lst = list()
        for svr in self.svr_lst:
            svr.set_CN(None)
        self.make_CN()



class CooperativeNet:   # Cluster (=Sub-region)
    def __init__(self, num_svr=1, ctrl=None):
        self.svr_lst = np.zeros(num_svr, dtype=np.bool_)
        self.ctrl = ctrl

    def add_svr(self, svr):
        if type(svr) == int:
            self.svr_lst[svr] = True
            # self.ctrl.svr_lst[svr].sub_region = self

        elif type(svr) == list:
            for idx in svr:
                self.svr_lst[idx] = True
                # self.ctrl.svr_lst[idx].sub_region = self
        else:
            raise Exception("Func add_svr: Wrong type error -> %s can not handle in this function" %type(svr))


    def isContain(self, data):
        if type(data) == int:
            data_idx = data
        elif isinstance(data, Data):
            data_idx = data.id
        else:
            raise Exception("Func isContain: type %s cannot be handled" % type(data))

        return self.ctrl.check_cache_in_CN(self.svr_lst, data_idx)


    def shortest_cache(self, svr, data):
        cache_lst = self.ctrl.caching_map[:, data.id]
        cache_lst = cache_lst * self.svr_lst
        rtt_lst = self.ctrl.rtt_map[svr.id, :]

        if np.any(rtt_lst > 0):
            min_rtt = np.min(rtt_lst[np.where(cache_lst > 0)])
            min_lst = np.where(rtt_lst == min_rtt)
            return np.random.choice(min_lst[0]), min_rtt
        else:
            return svr.id, 0


    def printCN(self):
        print("There are {} servers in cooperaitve network {}".format(self.svr_lst.tolist(), self))
        print("==============================================")

    '''
    def caching_policy(self):
        steps = []
        all_costs = []
        for episode in range(num_episodes):
            observation = self.env.reset()  # 환경 초기화
            step = 0
            cost = 0
            episode_reward = 0

            while True:
                # print("episode:", episode, "step:", step)
                s = self.env.states.index(observation['state'])
                # print("State: ", observation['state'])

                action = self._act(s)
                # print("action =", action)
                observation_next, reward, done = self.env.step(action)

                s_next = self.env.states.index(observation_next['state'])
                cost += self._update_q_value(s, action, reward, s_next, eta)

                # Swap observation
                observation = observation_next

                step += 1
                episode_reward += reward

                if done:
                    steps += [step]
                    all_costs += [cost]
                    break

            print("Episode: {} / {}, reward: {}, cost: {}".format(episode + 1, num_episodes, episode_reward, cost))

        result = self.env.get_cache_mat

        return result


    # For q-learning
    def _update_q_value(self, observation, action, reward, observation_next, alpha):
        return self.policy.update_q_value(observation, action, reward, observation_next, alpha)

    def _act(self, state):
        return self.policy.act(state)
    '''


class Cloud:
    def __init__(self, rtt, bandwidth):
        if type(rtt) != timedelta:
            self.rtt = timedelta(seconds=rtt)
        else:
            self.rtt = rtt

        self.bandwidth = bandwidth

    def calc_rtt(self, _):
        return self.rtt

    def append_request(self, **kwargs):
        data = kwargs['data']
        simulator = kwargs['simulator']
        processing_t = data.size / self.bandwidth
        processing_t = timedelta(seconds=processing_t)

        next_obj = simulator.find_target_obj(kwargs['event_path'][kwargs['path_idx']][0], kwargs['source'], kwargs['data'])
        input_value = {'start': False, 'source': kwargs['destination'], 'destination': next_obj}

        kwargs.update(input_value)
        simulator.make_event(simulator.T + processing_t, **kwargs)



