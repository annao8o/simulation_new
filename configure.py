import argparse
from enum import Enum, auto

# num_type = 5  # the number of clusters
# contents_num = 20
# z_val = 0.7
# user_num = 100
# cell_num = 5
# cache_capacity = 1
#
# bandwidth_back = 1 #Mbps
# bandwidth_downlink = 5 #MHz

# ## Zipf's skewness parameter
# skew_global_popularity = 0.6
# skew_activity_level = 0.4
# skew_location_prob = 1.0

class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Input environment parameters")
        self.parser.add_argument('-s', '--num_server', default=10, type=int)
        self.parser.add_argument('-u', '--num_user', default=100, type=int)
        self.parser.add_argument('-d', '--num_data', default=100, type=int)
        self.parser.add_argument('-t', '--num_type', default=5, type=int, help="The number of user types")
        self.parser.add_argument('-et', '--end_time', default=1000, type=int, help='End time (unit: seconds)')
        self.parser.add_argument('-a', '--arrival', default=1, type=float, help='Arrival rate of MEC server (unit: per seconds)')
        self.parser.add_argument('-c', '--cloud_rtt', default=0.1, type=float)
        self.parser.add_argument('-r', '--rtt', default=0.005, type=float)
        self.parser.add_argument('-b', '--bandwidth', default=200 * 1024 * 1024, type=int, help='bandwidth of MEC server (unit: bps)')
        self.parser.add_argument('-cb', '--cloud_bandwidth', default=1024*1024, type=int, help='bandwidth of cloud (unit: bps)')
        self.parser.add_argument('-zf', '--zipf_preference', default=0.7, type=float, help='zipf skewness parameter of user preference')
        self.parser.add_argument('-za', '--zipf_activity', default=0.4, type=float,
                                 help='zipf skewness parameter of user activity level')
        self.parser.add_argument('-zl', '--zipf_location', default=1.0, type=float,
                                 help='zipf skewness parameter of location probability')
        self.parser.add_argument('-f', '--file_size', nargs=2, default=[1000 * 1024 * 8, 1200 * 1024 * 8], type=int, help='file size')
        self.parser.add_argument('-cs', '--cache_size', default=6 * 1024 * 1024 * 8, type=int, help='cache size')

        self.parser.add_argument('--save_flag', action='store_true')
        self.parser.add_argument('--load_flag', action='store_false')
        self.parser.add_argument('-ig', '--init_graph', action='store_true')
        self.parser.add_argument('-ir', '--init_request', action='store_true')
        self.parser.add_argument('-lc', '--load_cache', action='store_false')
        self.parser.add_argument('-id', '--init_data', action='store_true')

        self.parser.add_argument('-sp', '--save_path', type=str, default='./save/')
        self.parser.add_argument('-lp', '--load_path', type=str, default='./save/')
        self.parser.add_argument('-op', '--output_path', type=str, default=None,
                                 help='output file path (None: not save output)')


class NetworkElement(Enum):
    user = auto()
    MECsvr = auto()
    cn = auto() # cooperative network
    cloud = auto()


class EventType(Enum):
    start = auto()
    end = auto()
    request = auto()
    process = auto()
    caching = auto()
    clustering = auto()
    path_check = auto()
    null = auto()


# (destination, type)
# type -> 0: start, 1: end, 2: waiting, -1: other
directfind = True
event_path_hit = [
    (NetworkElement.user, EventType.start),
    (NetworkElement.user, EventType.path_check),
    (NetworkElement.MECsvr, EventType.request),
    (NetworkElement.MECsvr, EventType.process),
    (NetworkElement.user, EventType.end)
]

event_path_miss = [
    (NetworkElement.user, EventType.start),
    (NetworkElement.user, EventType.path_check),
    (NetworkElement.cloud, EventType.request),
    (NetworkElement.cloud, EventType.process),   # transfer data
    (NetworkElement.user, EventType.end)
]

sim_theta = 0.7
split_point = 1   # split of path
user_rtt = (0.001, 0.002)
cache_capacity = 6 * 1024 * 1024 * 8
request_file = 'requests.bin'
integrated_file = 'integrated.bin'
caching_file = 'caching_map.bin'