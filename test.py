from genPreference import *
from elements import *
from configure import *
import numpy as np
from simulator import *
import os
import pickle


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


if __name__ == "__main__":
    parser = ArgsParser()
    args = parser.parser.parse_args()
    env = {
        'num server': args.num_server,
        'num user': args.num_user,
        'num data': args.num_data,
        'end time': timedelta(seconds=args.end_time),
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
        z = load_data['zipf']
        data_lst = load_data['data list']
        cn = load_data['cooperative network']
        request_lst = load_data['request list']

    else:   # initial generation
        args.init_graph = True
        args.init_data = True

    if args.init_data:
        generator = GenPreference()
        type_lst = generator.make_pref_type(args.num_type, args.num_data, args.zipf_preference)




    if args.init_graph:
        ctrl = Controller()
        ctrl.set_env(env['num server'], env['num data'], {'request rate': env['arrival rate']})
        ctrl.create_env(data_lst)
        ctrl.rtt_mapping()
        ctrl.make_cluster()
        args.init_cache = True
        args.init_request = True

    if args.init_request:
        request_lst = make_request_event(
            env['num server'], env['arrival rate'], 60, env['end time'], z, data_lst, ctrl.svr_lst
        )







    user_lst = []
    A = np.zeros((user_num, cell_num))
    Q = np.zeros((user_num, contents_num))

    ctrl = Controller()
    ctrl.create_env(data_lst, cell_num, cache_capacity)

    global_popularity = get_zipfs_distribution(contents_num, skew_global_popularity)
    activity_level = get_zipfs_distribution(user_num, skew_activity_level)
    location_prob = genLocality(num_type, cell_num, skew_location_prob)

    # Create users
    for i in range(user_num):
        u = User(i)
        user_type, pdf, cdf = generator.make_user_pref(dev_val=0.0001)
        u.set_char(user_type, pdf, cdf, activity_level[i], location_prob[user_type])
        user_lst.append(u)

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



