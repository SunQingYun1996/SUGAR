import numpy as np
import pandas as pd
from env import *


class QLearningTable:
    def __init__(self, actions, learning_rate=0.85, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if not done:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

def isTerminal(k_record):
    if len(k_record) >= 300:
        reward_sum = 0
        for ele in range(len(reward_record) - 10, len(reward_record)):
            reward_sum += reward_record[ele]
        if reward_sum <= 0.02:
            return True
        else:
            return False
    else:
        return False


def run_QL(env, RL, net, test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, initial_acc):
    step = 0
    reward_record = []
    k_record = []
    observation = env.reset(initial_acc)
    while True:
        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        # RL take action and get next observation and reward
        observation_, reward, done = env.step(action, observation, net, test_x, test_dsi, test_sadj, test_t, test_t_mi,
                                              test_mask, initial_acc)
        k_record.append(round(env.k, 2))
        reward_record.append(reward)
        # RL learn from this transition
        RL.learn(str(observation), action, reward, str(observation_), done)

        # swap observation
        observation = observation_
        step += 1
        if step % 100 == 0:
            print(step, reward_record)
            reward_record = []
        if isTerminal(reward_record):
            print('RL Terminal in ', step, 'step')
            print(k_record)
            return env.k




if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
