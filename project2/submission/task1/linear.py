# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Project 2 - Task 1
# Tran Hong Hai

import numpy as np
import matplotlib.pyplot as plt
from qlearning import *

# Number of states is 21, with two actions of going left (action 0) or going right (action 1)
class linear_environment:
    def __init__(self):
        self.n_states = 21       # number of states
        self.n_actions = 2      # number of actions
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype = np.int) # next_state[state, action]
        for i in range(self.n_states):
            if (i == 0) or (i == 20):
                self.next_state[i] = i
                continue
            self.next_state[i,0] = i - 1
            self.next_state[i,1] = i + 1
        self.reward = np.zeros([self.n_states, self.n_actions])     # reward[state, action]
        self.reward[1,0] = self.reward[19,1] = 1                    # set reward
        self.terminal = np.zeros(self.n_states, dtype = np.int)     # 1 if terminal state, 0 otherwise
        self.terminal[0] = self.terminal[self.n_states-1] = 1       # set terminal states
        self.init_state = 10     # initial state

# an instance of the environment
env = linear_environment()

# Epsilon profile class
class epsilon_profile: pass

############################################
# Epsilon = 1 (random walk), n_episodes = 1
############################################
print 'Epsilon = 1 (random walk), n_episodes = 1:'
n_episodes = 1      # number of episodes to run
max_steps = 1000    # max number of steps to run in each episode
alpha = 0.2         # learning rate
gamma = 0.9         # discount factor

epsilon = epsilon_profile()
epsilon.init = 1.    # initial epsilon in e-greedy
epsilon.final = 1.   # final epsilon in e-greedy
epsilon.dec_episode = 0.    # amount of decrement in each episode
epsilon.dec_step = 0.       # amount of decrement in each step

Q, n_steps, sum_rewards = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon)
print 'Q(s,a):'
print Q

test_n_episodes = 1     # number of episodes to run
test_max_steps = 1000   # max number of steps to run in each episode
test_epsilon = 0.       # test epsilon
test_n_steps, test_sum_rewards, s, a, sn, r = Q_test(Q, env, test_n_episodes, test_max_steps, test_epsilon)
print 'test_sum_rewards is', test_sum_rewards[0]
print 'test_n_steps is', test_n_steps[0]

############################################
# Epsilon = 1 (random walk), n_episodes = 5
############################################
print '\nEpsilon = 1 (random walk), n_episodes = 5:'
n_episodes = 5  # number of episodes to run

Q, n_steps, sum_rewards = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon)
print 'Q(s,a):'
print Q

test_n_steps, test_sum_rewards, s, a, sn, r = Q_test(Q, env, test_n_episodes, test_max_steps, test_epsilon)
print 'test_sum_rewards is', test_sum_rewards[0]
print 'test_n_steps is', test_n_steps[0]

###############################################
# Epsilon = 1 (random walk), n_episodes = 1000
###############################################
print '\nEpsilon = 1 (random walk), n_episodes = 1000:'
n_episodes = 1000  # number of episodes to run

Q, n_steps, sum_rewards = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon)
print 'Q(s,a):'
print Q
print 'Mean of n_steps is', np.mean(n_steps)

test_n_steps, test_sum_rewards, s, a, sn, r = Q_test(Q, env, test_n_episodes, test_max_steps, test_epsilon)
print 'test_sum_rewards is', test_sum_rewards[0]
print 'test_n_steps is', test_n_steps[0]


########################################################
# Epsilon is gradually decreased to 0, n_episodes = 100
########################################################
print '\nEpsilon is gradually decreased to 0, n_episodes = 100:'
n_episodes = 100    # number of episodes to run
max_steps = 1000    # max number of steps to run in each episode
alpha = 0.2         # learning rate
gamma = 0.9         # discount factor

epsilon.init = 1.    # initial epsilon in e-greedy
epsilon.final = 0.   # final epsilon in e-greedy
epsilon.dec_episode = 1./n_episodes     # amount of decrement in each episode
epsilon.dec_step = 0.                   # amount of decrement in each step

Q, n_steps, sum_rewards = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon)
print 'Q(s,a):'
print Q

test_n_episodes = 1     # number of episodes to run
test_max_steps = 1000   # max number of steps to run in each episode
test_epsilon = 0.       # test epsilon
test_n_steps, test_sum_rewards, s, a, sn, r = Q_test(Q, env, test_n_episodes, test_max_steps, test_epsilon)
print 'test_sum_rewards is', test_sum_rewards[0]
print 'test_n_steps is', test_n_steps[0]

plt.figure(1)
x_plot = np.arange(1, n_episodes+1)
plt.plot(x_plot, n_steps)
plt.grid(True)
plt.xlim(0, n_episodes+1)
plt.ylim(ymin = 0)
plt.xlabel('n_episodes')
plt.ylabel('n_steps')
plt.show()
