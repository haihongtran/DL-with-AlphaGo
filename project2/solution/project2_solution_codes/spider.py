# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/12/2017

import matplotlib.pyplot as plt
import numpy as np
from qlearning import *
from spider_ani import *
from spider_env import *
from wait import *

env = spider_environment()

n_episodes = 1         # number of episodes to run, 1 for continuing task
max_steps = 10000000   # max. # of steps to run in each episode
alpha = 0.1            # learning rate
gamma = 0.9            # discount factor

class epsilon_profile: pass
epsilon = epsilon_profile()
epsilon.init = 1.    # initial epsilon in e-greedy
epsilon.final = 0.   # final epsilon in e-greedy
epsilon.dec_episode = 0.  # amount of decrement in each episode
epsilon.dec_step = 1. / max_steps   # amount of decrement in each step

Q, n_steps, sum_rewards = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon)
print(sum_rewards[0])

test_n_episodes = 1  # number of episodes to run, 1 for continuing task
test_max_steps = 20  # max. # of steps to run in each episode
test_epsilon = 0.    # test epsilon
test_n_steps, test_sum_rewards, s, a, sn, r = Q_test(Q, env, test_n_episodes, test_max_steps, test_epsilon)
for j in range(test_max_steps):
    print('%10s %10s %10s %5.2f' % (bin(s[0,j]), bin(a[0,j]), bin(sn[0,j]), r[0,j]))
print(test_sum_rewards[0])

ani = spider_animation(Q, env, test_max_steps, test_epsilon)
# uncomment the following to save animation as *.mp4 (don't do plt.show() in that case since otherwise animation will run one more time with possibly different outcome)
#ani.save('spider.mp4', dpi=200)
plt.show(block=False)
wait('Press enter to quit')

