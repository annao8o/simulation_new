from genPreference import *
from elements import *
from configure import *
import numpy as np
from simulator import *
import os
import pickle
from request import make_request_events
from cacheAlgo import CacheAlgo


def get_zipfs_distribution(num, z_val):
    dist = np.zeros(num, dtype=np.float_)

    temp = np.power(np.arange(1, num + 1), -z_val)
    denominator = np.sum(temp)
    zipfs = temp / denominator

    unused_lst = [i for i in range(num)]
    np.random.shuffle(unused_lst)
    for i in range(num):
        dist[unused_lst[i]] = zipfs[i]

    return dist

def simulation(ctrl, s, request_lst):
    for svr in ctrl.svr_lst:
        zeta = np.r_[0.0, np.cumsum(svr.popularity)]
        cdf = [x / zeta[-1] for x in zeta]
        print("server id:{}\ncdf:{}".format(svr.id, cdf))
    # for svr in ctrl.svr_lst:
    #     sorted_lst = list(np.argsort(svr.popularity))
    #     ranking_lst = list()
    #     for f in ctrl.data_lst:
    #         ranking_lst.append(sorted_lst.index(f.id))
    #     print("server id:{}\nranking:{}".format(svr.id, ranking_lst))

    total_request = 0
    # total_hit = [[] for _ in range(len(ctrl.svr_lst))]
    # total_delay = [[] for _ in range(len(ctrl.svr_lst))]

    # ctrl.init_caching(y=y)

    # for ctrl in cloud.ctrl_lst:
    algo1 = CacheAlgo('proposed', ctrl.rtt_map, env)
    algo2 = CacheAlgo('greedy', ctrl.rtt_map, env)
    ctrl.add_algo(algo1)
    #ctrl.add_algo(algo2)

    total_hit = [0 for _ in range(len(ctrl.algo_lst))]
    total_delay = [0 for _ in range(len(ctrl.algo_lst))]

    s.init(ctrl)
    front_idx = 0
    t = 0
    print(request_lst[-1])

    for algo in ctrl.algo_lst:
        algo.init_caching(y)

    while t < env['end time']:
        #if t % update_period == 0:
            # ctrl.make_CN()
        #ctrl.init_caching(y)

        while front_idx < len(request_lst) and request_lst[front_idx][1] <= t:
            total_request += 1
            # print(t, request_lst[front_idx][1])
            svr = request_lst[front_idx][0]
            data = request_lst[front_idx][2]
            hit_lst, delay_lst = ctrl.add_request(RequestedData(data, t, svr))
            total_hit = [sum(x) for x in zip(total_hit, hit_lst)]
            total_delay = [sum(x) for x in zip(total_delay, delay_lst)]
            front_idx += 1
        t += 1

    print(total_request)
    print(total_hit)
    print(total_delay)


if __name__ == "__main__":
    parser = ArgsParser()
    args = parser.parser.parse_args()
    env = {
        'num server': args.num_server,
        'num user': args.num_user,
        'num data': args.num_data,
        'num type': args.num_type,
        'cache size': args.cache_size,
        'end time': args.end_time,
        'arrival rate': args.arrival,
        'cloud rtt': args.cloud_rtt,
        'rtt': args.rtt,
        'bandwidth': args.bandwidth,
        'cloud bandwidth': args.cloud_bandwidth,
        'zipf preference': args.zipf_preference,
        'zipf activity': args.zipf_activity,
        'zipf location': args.zipf_location
    }
    s = Simulator()
    s.set_env(env)  # generate cloud

    if args.load_flag:
        with open(os.path.join(args.load_path, integrated_file), 'rb') as f:
            load_data = pickle.load(f)
            print("success to load the integrated file")
        # z = load_data['zipf']
        data_lst = load_data['data list']
        ctrl = load_data['controller']
        request_lst = load_data['request list']
        args.init_graph = False
        args.init_data = False
        args.init_request = False
        args.load_cache = True

    else:   # initial generation
        args.init_graph = True
        args.init_data = True
        args.save_flag = True

    if args.init_data:
        data_lst = [Data(i, size=random.randint(args.file_size[0], args.file_size[1])) for i in range(env['num data'])]  # data 생성
        activity_level = get_zipfs_distribution(env['num user'], env['zipf activity'])
        location_prob = genLocality(env['num type'], env['num server'], env['zipf location'])

        generator = GenPreference()
        generator.set_env(env['num type'], env['num data'], env['zipf preference'])
        user_lst = generator.make_user(env['num user'], activity_level, location_prob)
        A = np.zeros((env['num user'], env['num server']))
        Q = np.zeros((env['num user'], env['num data']))
        for u in user_lst:
            # u.print_user_info()
            A[u.id] = u.location
            Q[u.id] = u.pref_vec[0]
        local = calc_local_popularity(Q, A, activity_level)

        args.init_request = True

    if args.init_graph:
        ctrl = Controller()
        ctrl.set_env(env['num server'], env['num data'], {'request rate': env['arrival rate']})
        ctrl.create_env(data_lst)
        ctrl.rtt_mapping()
        ctrl.set_popularity(local)
        ctrl.make_CN(sim_theta)

    if args.init_request:
        request_lst = make_request_events(
            env['num server'], env['arrival rate'], 60, env['end time'], data_lst, ctrl.svr_lst
        )

    if args.save_flag:
        with open(os.path.join(args.save_path, integrated_file), 'wb') as f:
            save_data = {'network_env': env, 'data list': data_lst, 'user list': user_lst, 'controller': ctrl, 'request list': request_lst}
            pickle.dump(save_data, f)
            print("success to save")

    # load the caching map file
    if args.load_cache:
        with open(os.path.join(args.load_path, caching_file), 'rb') as f:
            y = pickle.load(f)
            print("success to load the caching file")

        simulation(ctrl=ctrl, s=s, request_lst=request_lst)

    # if args.output_path is not None:
    #     with open(args.output_path, 'wb') as f:
    #         pickle.dump(s.result, f)
    #


'''
    type_lst = generator.make_pref_type(args.num_type, args.num_data, args.zipf_preference)

    user_lst = []
    A = np.zeros((user_num, cell_num))
    Q = np.zeros((user_num, contents_num))

    # ctrl = Controller()
    # ctrl.create_env(data_lst, cell_num, cache_capacity)
    #
    # global_popularity = get_zipfs_distribution(contents_num, skew_global_popularity)
    # activity_level = get_zipfs_distribution(user_num, skew_activity_level)
    # location_prob = genLocality(num_type, cell_num, skew_location_prob)

    # # Create users
    # for i in range(user_num):
    #     u = User(i)
    #     user_type, pdf, cdf = generator.make_user_pref(dev_val=0.0001)
    #     u.set_char(user_type, pdf, cdf, activity_level[i], location_prob[user_type])
    #     user_lst.append(u)

    # Set user's character
    for u in user_lst:
         # u.print_user_info()
        A[u.id] = u.location
        Q[u.id] = u.pref_vec[0]

    # Set popularity of each cell
    local = calc_local_popularity(Q, A, activity_level)
    for i in range(len(local)):
        ctrl.svr_lst[i].set_popularity(local[i, :])


    print("Global\n",calc_global_popularity(Q, activity_level))
    print("Local\n", local)
'''


