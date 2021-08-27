from genPreference import *
from elements import *
from configure import *
import numpy as np
from simulator import *
import os
import pickle
from request import make_request_events


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
        'num type': args.num_type,
        'cache size': args.cache_size,
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
            print("success to load the integrated file")
        # z = load_data['zipf']
        data_lst = load_data['data list']
        ctrl = load_data['controller']
        request_lst = load_data['request list']
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
        ctrl.init_caching(y=y)

        s.init(ctrl)
        chk_length = 1000
        buff_time = timedelta(seconds=60)
        front_idx = 0
        rear_idx = 1

        state = 0
        front_t = timedelta(seconds=0)

        while state == 0:
            if front_t < s.T + buff_time and front_idx < len(request_lst):
                while rear_idx < len(request_lst) and request_lst[rear_idx - 1][1] < s.T + buff_time:   #request_lst[][1]: time
                    rear_idx += chk_length
                    rear_idx = rear_idx if rear_idx < len(request_lst) else len(request_lst)
                s.insert_request_lst(request_lst[front_idx:rear_idx])
                front_idx = rear_idx
                front_idx = front_idx if front_idx < len(request_lst) else len(request_lst)
                front_t = request_lst[front_idx][1] if front_idx < len(request_lst) else request_lst[-1][1]
            state = s.update()

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


