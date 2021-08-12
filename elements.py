import numpy as np
import networkx as nx
from itertools import combinations
import random
from datetime import timedelta
from collections import deque



class Data:
    def __init__(self, i, data_size):
        self.id = i
        self.size = data_size

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
    def __init__(self, id, capacity, controller):
        self.id = id
        self.ctrl = controller
        self.location = None
        self.capacity = capacity
        self.storage_usage = 0
        self.queue = deque()
        self.popularity = None
        self.service_time = 0
        self.CN = None

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

    def set_CN(self, sub):
        self.sub_region = sub

    def check_CN(self, data):    #check whether the data is stored in the sub-region
        return self.sub_region.isContain(data)

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

    def calc_service_time(self ):



    def append_request(self, **kwargs):
        kwargs['data'].set_queueing(kwargs['simulator'].T)
        self.queue.append(kwargs)
        if self.processing_event is None:
            self.processing_end()
        if len(self.queue) == 1:
            return 1
        else:
            return 0



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


    def set_env(self, num_svr, num_data):
        self.num_svr = num_svr
        self.num_data = num_data
        self.caching_map = np.zeros((num_svr, num_data), dtype=np.bool_)
        self.d_size_map = np.zeros(num_data, dtype=np.int_)
        # self.requests = requests


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


    def set_svr_cache(self, svr_idx, item):
        if type(item) == np.ndarray:
            self.caching_m

    def check_cache_in_svr(self, svr_idx, data_idx):
        return self.caching_map[svr_idx, data_idx]

    def check_cache_in_CN(self, CN_matrix, data_idx):
        return self.caching_map[:, data_idx]@CN_matrix

    def make_CN(self):
        checkList = [i for i in range(self.num_svr)]
        while checkList:
            cn = CooperativeNet(self.num_svr, self)
            i = random.choice(checkList)
            checkList.remove(i)
            cn.add_svr(i)
            self.svr_lst[i].set_CN(cn)

            for n in self.graph.neighbors(i):
                if n in checkList:
                    # 클러스터 기준 다시 정해서 코딩해야함

    def rtt_mapping(self):
        self.rtt_map = np.zeros((self.num_svr, self.num_svr), dtype=np.float_)
        for i in range(self.num_svr):
            for j in range(i+1, self.num_svr):
                length = nx.shortest_path_length(self.graph, i, j, weight='rtt')
                self.rtt_map[i, j] = length
                self.rtt_map[j, i] = length





class CooperativeNet:   # Cluster (=Sub-region)
    def __init__(self, num_svr=1, ctrl=None):
        self.svr_lst = np.zeros(num_svr, dtype=np.bool_)
        self.ctrl = ctrl

    def add_svr(self, svr):
        if type(svr) == int:
            self.svr_lst[svr] = True
            self.ctrl.svr_lst[svr].sub_region = self

        elif type(svr) == list:
            for idx in svr:
                self.svr_lst[idx] = True
                self.ctrl.svr_lst[idx].sub_region = self

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





class Cloud:
    def __init__(self, rtt, bandwidth):
        if type(rtt) != timedelta:
            self.rtt = timedelta(seconds=rtt)
        else:
            self.rtt = rtt

        self.bandwidth = bandwidth

    def calc_rtt(self):
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



