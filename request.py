from datetime import timedelta
import numpy as np
from configure import request_file
import pickle
import os


def make_request_events(num_svr, arrival_rate, interval, end_t, data_lst, svr_lst):
    request_events = list()
    t = timedelta(0)
    t_interval = timedelta(seconds=interval)

    while t < end_t:
        req_lst = np.random.poisson(arrival_rate * interval, size=num_svr)
        for svr_idx in range(num_svr):
            if req_lst[svr_idx] != 0:
                r_interval = t_interval / req_lst[svr_idx]
                for i in range(req_lst[svr_idx]):
                    e_t = t + r_interval / 2 + r_interval * i
                    request_events.append((svr_lst[svr_idx], e_t, data_lst[svr_lst[svr_idx].get_sample()]))
        t += t_interval
    request_events.sort(key=lambda x: x[1])
    return request_events


def load_request_events(folder_path):
    with open(os.path.join(folder_path, request_file), 'rb') as f:
        return pickle.load(f)


def save_request_events(folder_path, request_events):
    from configure import request_file
    with open(os.path.join(folder_path, request_file), 'wb') as f:
        pickle.dump(request_events, f)
