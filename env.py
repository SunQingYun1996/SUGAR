import numpy as np


class GNN_env(object):
    def __init__(self):
        super(GNN_env, self).__init__()
        self.action_space = ['add', 'minus']
        self.action_value = 0.05
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.k = 0.5

    def reset(self, initial_acc):
        initial_observation = [self.k, initial_acc]
        initial_observation = np.array(initial_observation)
        return initial_observation

    def step(self, action, last_observation, net, test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask,
             initial_acc):
        if action == 0:  # add
            if self.k <= 0.95:
                self.k += 0.05
        elif action == 1:  # minus
            if self.k >= 0.05:
                self.k -= 0.05
        eva_loss, eva_acc, _, _ = net.evaluate(test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, self.k)
        eva_acc = round(eva_acc, 3)
        s_ = [self.k, eva_acc]
        s_ = np.array(s_)
        if s_[-1] > last_observation[-1]:
            reward = 1
            done = True
        elif s_[-1] < last_observation[-1]:
            reward = -1
            done = True
        else:
            reward = 0
            done = True
        return s_, reward, done

    def render(self):
        self.update()
