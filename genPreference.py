import numpy as np
import random
from elements import User

def calc_global_popularity(Q, v):
    user_num = len(Q)
    file_num = len(Q[0])
    p = np.zeros(file_num)

    for f in range(file_num):
        p_f = 0
        for u in range(user_num):
            p_f += v[u] * Q[u][f]
        p[f] = p_f
    # print(p.argsort())
    return p


def calc_local_popularity(Q, A, v):
    user_num = len(Q)
    file_num = len(Q[0])
    cell_num = len(A[0])
    p_i = np.zeros(file_num)
    local_p = np.zeros((cell_num, file_num))


    for i in range(cell_num):
        for f in range(file_num):
            denominator = 0
            numerator = 0
            for u in range(user_num):
                denominator += A[u][i] * v[u]
                numerator += A[u][i] * v[u] * Q[u][f]
            p_fi = numerator / denominator
            p_i[f] = p_fi
        local_p[i] = p_i
        # print(local_p[i].argsort())

    return local_p


def genLocality(num_type, cell_num, z_val):
    locality = np.zeros((num_type, cell_num), dtype=np.float_)

    temp = np.power(np.arange(1, cell_num + 1), -z_val)
    denominator = np.sum(temp)
    pdf = temp / denominator

    h_lst = [i for i in range(cell_num)]
    bound = cell_num // num_type
    for type_idx in range(num_type):
        unused_lst = [i for i in range(cell_num)]
        for i in range(bound):
            c_i = np.random.choice(h_lst)
            h_lst.remove(c_i)
            unused_lst.remove(c_i)
            locality[type_idx, c_i] = pdf[i]
        np.random.shuffle(unused_lst)
        for i in range(bound, cell_num):
            locality[type_idx, unused_lst[i - bound]] = pdf[i]

    return locality


def make_requests(svr_lst, arrival_rate, interval, total_time):
   req_lst = list()
   t = 0
   while t < total_time:
       req_num = np.random.poisson(arrival_rate, size = len(svr_lst))
       for s in svr_lst:
           # cnt_dict = dict()
           if req_num[s.id] != 0:
               n = req_num[s.id]
               samples = get_sample(n, s.popularity)
               for sample in samples:
                   # cnt_dict[sample] = cnt_dict.get(sample, 0)+1
                   req_lst.append((t, s.id, sample)) #(t, server_id, content)
           # a = sorted(cnt_dict.items(), key=lambda x: x[1], reverse=True)
       t += interval
   req_lst.sort(key=lambda x: x[0])
   return req_lst


def get_sample(size, popularity):
    zeta = np.r_[0.0, np.cumsum(popularity)]
    cdf = [x / zeta[-1] for x in zeta]

    if size is None:
        f = random.random()
    else:
        f = np.random.random(size)
    v = np.searchsorted(cdf, f)
    samples = [t - 1 for t in v]
    return samples


class GenPreference:
    def __init__(self):
        self.type_lst = None  # store preference type
        self.num_type = None
        self.num_data = None
        self.z_val = None
        self.user_lst = list()


    def set_env(self, num_type, num_data, z_val):
        self.num_type = num_type
        self.num_data = num_data
        self.z_val = z_val
        self.type_lst = np.zeros((num_type, num_data), dtype=np.float_)


    def make_pref_type(self):
        temp = np.power(np.arange(1, self.num_data + 1), -self.z_val)
        denominator = np.sum(temp)
        pdf = temp / denominator

        h_lst = [i for i in range(self.num_data)]
        bound = self.num_data // self.num_type
        for type_idx in range(self.num_type):
            unused_lst = [i for i in range(self.num_data)]
            for i in range(bound):
                c_i = np.random.choice(h_lst)
                h_lst.remove(c_i)
                unused_lst.remove(c_i)
                self.type_lst[type_idx, c_i] = pdf[i]
            np.random.shuffle(unused_lst)
            for i in range(bound, self.num_data):
                self.type_lst[type_idx, unused_lst[i - bound]] = pdf[i]


    def make_user(self, user_num, activity_level, location_prob):
        for i in range(user_num):
            u = User(i)
            user_type, pdf, cdf = self.make_user_pref(dev_val=0.0001)
            u.set_char(user_type, pdf, cdf, activity_level[i], location_prob[user_type])
            self.user_lst.append(u)

        return self.user_lst


    def make_user_pref(self, dev_val=0.001):  # type_weight: 타입과의 유사도? p = w1*t1 + w2*t2 + ...  , out_type: 0=cdf, 1=pdf
        # if type_weight is None:
        #     type_weight = np.random.random((self.num_type, 1))
        #
        # type_weight = type_weight / type_weight.sum()  # normalization
        #
        # # result = self.type_lst * type_weight
        # # result = result.sum(axis=0)    # pdf
        # user_type = np.random.choice(len(type_weight), p=type_weight.flatten())
        self.make_pref_type()
        user_type = np.random.choice(len(self.type_lst))
        result = self.add_noise(self.type_lst[user_type, :], dev_val)  # add noise for user

        return user_type, result, np.r_[0.0, np.cumsum(result)]  # return pdf, cdf


    def add_noise(self, values, noise_value):  # array에 noise를 더함
        if type(values) == list:
            values = np.array(values)

        values = values / values.sum()
        dev = np.random.random(values.shape)
        dev = dev - dev.sum() / len(dev)
        dev = dev * noise_value * 2  # dev 범위 = (-0.5, 0.5) -> (-noise_value, noise_value)

        result = values + dev  # add noise

        result[np.where(result < 0.0)] = 0.0
        result[np.where(result > 1.0)] = 1.0

        result = result / result.sum()  # normalize

        return result




