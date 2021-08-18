from datetime import timedelta
from collections import deque
from enum import Enum, auto
from elements import *
from copy import copy
import numpy as np
from configure import *


def event_func(time=None, **kwargs):
    path_info = kwargs['event_path'][kwargs['path_idx']]
    simulator = kwargs['simulator']
    return_v = {
        'start time': kwargs['data'].request_time,
        'type': path_info[1],
        'place': kwargs['source'],
        'total delay': simulator.T - kwargs['data'].request_time
    }

    if 'waiting delay' in kwargs:
        return_v['waiting delay'] = kwargs['waiting delay']

    if kwargs['next_idx'] >= 0:  # not end
        kwargs['path_idx'] = kwargs['next_idx']
        if path_info[1] == EventType.process:
            kwargs['source'].append_request(**kwargs)
        else:
            time = time + kwargs['source'].calc_rtt(kwargs['destination'])
            kwargs['source'] = kwargs['destination']
            kwargs['destination'] = simulator.find_target_obj(
                kwargs['event_path'][kwargs['next_idx']][0], kwargs['source'], kwargs['data']
            )
            simulator.make_event(time, **kwargs)
    return return_v


class Simulator:
    def __init__(self):
        self.result = None

        def r_f(simulator, result):
            if result is None:
                pass
            elif result['type'] == EventType.request:
                simulator.result['total_request'] += 1

            elif result['type'] == EventType.end:
                simulator.result['sum_delay'] += result['total delay']
                simulator.result['total_request_end'] += 1
                if 'waiting delay' in result:
                    simulator.result['sum_hit_delay'] += result['total delay']
                    simulator.result['sum_waiting_delay'] += result['waiting delay']
                    simulator.result['total_hit'] += 1

        self.result_func = r_f
        self.env = None
        self.T = None   # current time
        self.event_q = None
        self.end_time = None
        self.user = User
        self.cloud = None
        self.ctrl= None

    def set_env(self, env):
        self.env = env
        self.cloud = Cloud(env['cloud rtt'], env['cloud bandwidth'])

    def init(self, ctrl):
        self.T = timedelta(0)
        self.event_q = deque()
        self.end_time = copy(self.env['end time'])
        self.result = {'total_request': 0,
                       'sum_delay': timedelta(seconds=0),
                       'sum_hit_delay': timedelta(seconds=0),   #?????
                       'sum_waiting_delay': timedelta(seconds=0),
                       'total_hit': 0,
                       'total_request_end': 0
                       }
        self.ctrl = ctrl

    def insert_event(self, event):
        if len(self.event_q) > 0:
            event_time = event.time
            rotation = 0
            front_t = self.event_q[0].time
            rear_t = self.event_q[-1].time
            if event_time <= front_t:
                self.event_q.appendleft(event)
            elif rear_t <= event_time:
                self.event_q.append(event)
            else:
                def comp_func(event_t, event_queue, rotation_v):
                    if rotation_v == 1:  # clockwise
                        return event_queue[-1].time <= event_t
                    else:  # counter clockwise
                        return event_t <= event_queue[0].time

                if event_time - front_t <= rear_t - event_time:  # close to front: rotate counter clockwise
                    rotation_v = -1
                else:  # close to rear: rotate clockwise
                    rotation_v = 1

                for i in range(len(self.event_q)):
                    if comp_func(event_time, self.event_q, rotation_v):
                        if rotation_v == 1:
                            self.event_q.append(event)
                        else:
                            self.event_q.appendleft(event)
                        break
                    else:
                        self.event_q.rotate(rotation_v)
                        rotation += rotation_v

                if rotation == len(self.event_q):  # rotate all (compare all items are False)
                    if rotation_v == 1:  # reverse insertion
                        self.event_q.appendleft(event)
                    else:
                        self.event_q.append(event)

                else:  # recover order
                    self.event_q.rotate(-rotation)
        else:
            self.event_q.append(event)


    def make_event(self, time, **kwargs):
        path_info = kwargs['event_path'][kwargs['path_idx']]
        if path_info[1] == EventType.start:
            kwargs['source'] = self.find_target_obj(path_info[0])
        kwargs['destination'] = self.find_target_obj(path_info[0], kwargs['source'], kwargs['data'])

        event_path, next_idx = self.find_next_event(**kwargs)
        kwargs['event_path'] = event_path
        kwargs['next_idx'] = next_idx

        event = Event(self, time, event_func, kwargs, path_info[1])
        self.insert_event(event)

    def path_check(self, **kwargs):
        if kwargs['data'].near_svr.check_CN(kwargs['data']):
            return event_path_hit
        else:
            return event_path_miss

    def make_func_event(self, time, func, **kwargs):
        event = Event(self, time, func, kwargs)
        self.insert_event(event)

    def find_next_event(self, **kwargs):
        if kwargs['path_idx']+1 < len(kwargs['event_path']):    # not end
            next_element_info = kwargs['event_path'][kwargs['path_idx']]
            if next_element_info[1] == EventType.path_check:
                return self.path_check(**kwargs), split_point + 1
            else:
                return kwargs['event_path'], kwargs['path_idx'] + 1
        else:
            return kwargs['event_path'], -1



    def insert_request_lst(self, requests):
        for svr, t, data in requests:
            if type(t) != timedelta:
                t = timedelta(seconds=t)
            req_data = RequestedData(data, t, svr)
            # req_data = data

            self.make_event(t, source=self.user, destination=self.user, path_idx=0, event_path=event_path_hit, start=True, data=req_data, svr=svr)


    def make_process_event(self, svr, **kwargs):
        data = kwargs['data']
        processing_t = data.size / self.env['bandwidth']
        processing_t = timedelta(seconds=processing_t)

        next_obj = self.find_target_obj(kwargs['event_path'][kwargs['path_idx']][0], kwargs['source'], kwargs['data'])
        input_value = {
            'process start time': self.T, 'start': False, 'waiting delay': self.T - data.queueing_time,
            'source': kwargs['destination'], 'destination': next_obj
        }
        kwargs.update(input_value)

        self.make_func_event(self.T + processing_t, svr.processing_end, **kwargs)  # pop bc queue
        self.make_event(self.T + processing_t, **kwargs)


    def find_target_obj(self, target_element, source_obj=None, data_obj=None):
        if target_element == NetworkElement.user:
            return self.user

        elif target_element == NetworkElement.MECsvr:
            if source_obj is None or data_obj is None:
                raise Exception("need source and data object")
            type_source = type(source_obj)
            if User in (type_source, source_obj):
                if directfind:
                    near_svr = data_obj.near_svr
                    svr_idx, _ = near_svr.cn.shortest_cache(near_svr, data_obj)
                else:
                    svr_idx = data_obj.near_svr.id

            elif MECServer in (type_source, source_obj):
                svr_idx, _ = source_obj.cluster.shortest_cache(source_obj, data_obj)

            elif Cloud in (type_source, source_obj):
                svr_idx = data_obj.near_svr
            else:
                raise Exception("network element %s is not handled in this function" % source_obj)
            return self.ctrl.svr_lst[svr_idx]

        elif target_element == NetworkElement.cloud:
            return self.cloud

        else:
            raise Exception("network element %s is not handled in this function" % type(target_element))



    def update(self):   # running:0, end: 1, no event: -1
        if self.T <= self.end_time and self.event_q:
            event = self.event_q.popleft()
            self.T = event.time
            result = event.run()
            self.result_func(self, result)
            return 0
        elif self.event_q:
            return -1
        else:
            return 1


class Event:
    def __init__(self, simulator=None, time=None, func=None, func_input=None, event_type=EventType.null):
        self.time = time
        self.func = func
        self.input = func_input
        self.type = event_type
        func_input['simulator'] = simulator

    def run(self):
        return self.func(time=self.time, **self.input)
