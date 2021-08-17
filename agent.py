from policy import QlearningPolicy, GreedyPolicy
import matplotlib.pyplot as plt
import numpy as np

## Controller server
class Agent:
    def __init__(self, env, gamma, Q):
        self.server_list = list()
        self.policy = QlearningPolicy(env, gamma, Q)

    def set_cooperNet(self, svr):
        self.server_list.append(svr)

    def get_cooperNet(self):
        return self.server_list

    def _update_q_value(self, observation, action, reward, observation_next, alpha):
        return self.policy.update_q_value(observation, action, reward, observation_next, alpha)

    def _act(self, state):
        return self.policy.act(state)

    def plot_results(self, steps, cost):
        #
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        #
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        #
        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        plt.tight_layout()  # Function to make distance between figures

        #
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        #
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, 'r')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        # Showing the plots
        plt.show()


## MEC server:
class MECServer:
    def __init__(self, id, cache_size):
        self.id = id
        self.cache_size = cache_size
        self.cache = [-1] * self.cache_size
        self.wait_q = None

    def get_state(self):
        return self.cache

    def update_cache(self, file):
        if len([x for x in self.cache if x >= 0]) < self.cache_size:
            self.cache.append(file)
            print("Cache the file {} to server {}".format(file, self.id))
        else:
            print("Cache {} is already full.".format(self.id))





