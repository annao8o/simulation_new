import numpy as np
import os
import pickle

class DataLoader(object):
    def __init__(self):
        self.requests = []
        self.operations = []

    def get_requests(self):
        pass
    def get_operations(self):
        pass

class DataLoaderZipf(DataLoader):
    def __init__(self, num_files, num_samples, param, num_progs=1, operation='random'):
        super(DataLoaderZipf, self).__init__()

        for i in range(num_progs):
            files = np.arange(num_files)
            # Random ranks. Note that it starts from 1.
            ranks = np.random.permutation(files) + 1
            # Distribution
            pdf = 1 / np.power(ranks, param)
            pdf /= np.sum(pdf)
            # Draw samples
            self.requests.append(np.random.choice(files, size=num_samples, p=pdf).tolist())
            if operation == 'random':
                self.operations += np.random.choice([0, 1], size=num_samples).tolist()
            else:
                self.operations += np.full(num_samples, int(operation)).tolist()

    def get_requests(self):
        return self.requests

    def get_operations(self):
        return self.operations


class MakePopularity:
    def __init__(self, n_data, z_val):
        self.n_data = n_data
        self.z_val = z_val

        temp = np.power(np.arange(1, n_data+1), -z_val)
        denominator = np.sum(temp)
        self.pdf = temp / denominator

    def get_popularity(self, n_servers):
        popularity_mat = np.zeros((n_servers, self.n_data))
        for i in range(n_servers):
            np.random.shuffle(self.pdf)
            popularity_mat[i] = self.pdf

        return popularity_mat

def load_env_from_file(load_path, file):
    with open(os.path.join(load_path, file), 'rb') as f:
        load_data = pickle.load(f)
        print("success to load the integrated file")

    return load_data
