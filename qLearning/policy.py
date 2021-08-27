import numpy as np

class QlearningPolicy:
    def __init__(self, env, gamma=0.99, Q=None):
        """
        1. We start from state s and
        2. At state s, with action a, we observe a reward r(s, a) and get into the next state s'. Update Q function:
            Q(s, a) += learning_rate * (r(s, a) + gamma * max Q(s', .) - Q(s, a))
        Repeat this process.
        """
        self.env = env
        self.gamma = gamma
        self.Q = Q
        self.actions = range(self.env.n_actions)

    def act(self, state):
        """
        Pick the best action according to Q values ~ argmax_a Q(s, a).
        Exploration is forced by epsilon-greedy.
        """
        eps = 0.1
        # epsilon-greedy
        if np.random.uniform(0, 1) < eps:
            action = np.random.choice(self.env.actions)  # Explore action space
        else:
            action = np.argmax(self.Q[state])   # Exploit learned values

        return action

    def update_q_value(self, s, action, reward, s_next, eta):
        old_value = self.Q[s, action]
        next_max = np.max(self.Q[s_next])

        new_value = (1 - eta) * old_value + eta * (reward + self.gamma * next_max - old_value)
        self.Q[s, action] = new_value

        return self.Q[s, action]



class GreedyPolicy:
    def __init__(self, p_m, l, delta_0, delta, d_0_m, d_m_n, s):
        self.popualrity_map = p_m
        self.data_size = l
        self.delta_0 = delta_0
        self.delta = delta
        self.d_0_m = d_0_m
        self.d_m_n = d_m_n
        self.s = s


    def run_algorithm(self):
        mat_x = np.zeros(self.p_m.shape, dtype=np.bool_) # caching matrix

        while True:
            theta = self.calculate_theta(mat_x)

            if np.any(theta > 0):
                target_idx = np.unravel_index(np.argmax(theta), theta.shape)
                mat_x[target_idx] = 1
            else:
                break

        return mat_x

    def calculate_theta(self, mat_x):
        theta = np.zeros(self.mat_x.shape)
        h_f = np.sum(self.mat_x, axis=1) * self.l
        h_f = h_f + self.l

        for f in range(theta.shape[1]):
            if not mat_x[:, f].any():
                for m in np.where(h_f <= self.s)[0]:
                    if not mat_x[m, f]:
                        for n in range(theta.shape[0]):
                            theta[m, f] += ((self.p_m[n, f] * self.l) * (self.delta_0 * self.d_0_m[n] - self.delta * self.d_m_n[m, n]))

            elif not mat_x[:, f].all():
                for m in np.where(h_f <= self.s)[0]:
                    if not mat_x[m, f]:
                        for n in range(theta.shape[0]):
                            d_n_k = np.min(self.d_m_n[n, np.where(mat_x[:, f])])
                        theta[m, f] += ((self.p_m[n, f] * self.l * self.delta) * (d_n_k - self.d_m_n[m, n])) if d_n_k > self.d_m_n[m, n] else 0
            else:
                theta[:, f] = 0

        return theta