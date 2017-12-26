# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/18/2017

import numpy as np
from qlearning import *
import matplotlib.pyplot as plt

# environment with 21 states where 2 of them are terminal states (episodic task)
class linear_environment:
    def __init__(self):
        self.n_states = 21      # number of states
        self.n_actions = 2      # number of actions (0: left, 1: right)
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype=np.int)    # next_state
        for s in range(self.n_states):
            self.next_state[s, 0] = max(s - 1, 0)                                    # next state does not need to be defined for terminal states
            self.next_state[s, 1] = min(s + 1, self.n_states - 1)                    #   but we also define them anyway
        self.reward = np.zeros([self.n_states, self.n_actions])                      # reward for each (state,action)
        self.reward[1,0] = self.reward[-2,1] = 1.                                    # reward is 1 if reaching first (0) or last (20) state
        self.terminal = np.zeros(self.n_states, dtype=np.int)                        # 1 if terminal state, 0 otherwise
        self.terminal[0] = self.terminal[-1] = 1
        self.init_state = self.n_states // 2     # initial state (center)

# an instance of the environment
env = linear_environment()

n_episodes = 100     # number of episodes to run
max_steps = 1000     # max. # of steps to run in each episode
alpha = 0.2          # learning rate
gamma = 0.9          # discount factor

class epsilon_profile: pass
epsilon = epsilon_profile()
epsilon.init = 1.    # initial epsilon in e-greedy
epsilon.final = 0.   # final epsilon in e-greedy
epsilon.dec_episode = 1. / n_episodes  # amount of decrement in each episode
epsilon.dec_step = 0.                  # amount of decrement in each step

Q, n_steps, sum_rewards = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon)
print('Q(s,a)')
print(Q)
for k in range(n_episodes):
    print('%2d: %d' % (k, n_steps[k]))
print('Average number of runs: %.1f' % np.mean(n_steps))

test_n_episodes = 1     # number of episodes to run
test_max_steps = 1000   # max. # of steps to run in each episode
test_epsilon = 0.       # test epsilon
test_n_steps, test_sum_rewards, s, a, sn, r = Q_test(Q, env, test_n_episodes, test_max_steps, test_epsilon)
print('Number of steps in testing %d' % test_n_steps[0])

fig, ax = plt.subplots()
ax.plot(np.arange(100)+1., n_steps)
ax.grid()
ax.set(xlabel='episode', ylabel='# steps')
#fig.savefig('linear.png', dpi=200)
plt.show()

