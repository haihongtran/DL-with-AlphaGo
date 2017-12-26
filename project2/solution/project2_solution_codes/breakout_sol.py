# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/19/2017

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from breakout_env import *
from wait import *

env = breakout_environment(5, 8, 3, 1, 2)

c1 = 30
c2 = 30
f1 = 128

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, env.ny, env.nx, env.nf])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
a_ = tf.placeholder(tf.float32, shape=[None, env.na])

# First convolutional layer
W_conv1 = tf.Variable(tf.truncated_normal([3, 3, env.nf, c1], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[c1]))
h_conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_relu1 = tf.nn.relu(h_conv1 + b_conv1)

# Second convolutional layer
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, c1, c2], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[c2]))
h_conv2 = tf.nn.conv2d(h_relu1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_relu2 = tf.nn.relu(h_conv2 + b_conv2)

# Fully-connected Layer
W_fc1 = tf.Variable(tf.truncated_normal([env.ny * env.nx * c2, f1], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[f1]))
h_flat = tf.reshape(h_relu2, [-1, env.ny * env.nx * c2])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([f1, env.na], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[env.na]))
y_hat = tf.matmul(h_fc1, W_fc2) + b_fc2

# Cost function and optimizer
mse = tf.reduce_mean(tf.square(tf.reduce_sum(y_hat * a_, 1, keep_dims=True)- y_))
train_step = tf.train.AdamOptimizer(0.001).minimize(mse)

# First convolutional layer for target network
W_conv1_t = tf.Variable(tf.constant(0., shape=[3, 3, env.nf, c1]))
b_conv1_t = tf.Variable(tf.constant(0., shape=[c1]))
h_conv1_t = tf.nn.conv2d(x, W_conv1_t, strides=[1, 1, 1, 1], padding='SAME')
h_relu1_t = tf.nn.relu(h_conv1_t + b_conv1_t)

# Second convolutional layer for target network
W_conv2_t = tf.Variable(tf.constant(0., shape=[3, 3, c1, c2]))
b_conv2_t = tf.Variable(tf.constant(0., shape=[c2]))
h_conv2_t = tf.nn.conv2d(h_relu1_t, W_conv2_t, strides=[1, 1, 1, 1], padding='SAME')
h_relu2_t = tf.nn.relu(h_conv2_t + b_conv2_t)

# Fully-connected Layer for target network
W_fc1_t = tf.Variable(tf.constant(0., shape=[env.ny * env.nx * c2, f1]))
b_fc1_t = tf.Variable(tf.constant(0., shape=[f1]))
h_flat_t = tf.reshape(h_relu2_t, [-1, env.ny * env.nx * c2])
h_fc1_t = tf.nn.relu(tf.matmul(h_flat_t, W_fc1_t) + b_fc1_t)

# Output layer for target network
W_fc2_t = tf.Variable(tf.constant(0., shape=[f1, env.na]))
b_fc2_t = tf.Variable(tf.constant(0., shape=[env.na]))
y_hat_t = tf.matmul(h_fc1_t, W_fc2_t) + b_fc2_t


def update_target():
    sess.run(tf.assign(W_conv1_t, W_conv1))
    sess.run(tf.assign(b_conv1_t, b_conv1))
    sess.run(tf.assign(W_conv2_t, W_conv2))
    sess.run(tf.assign(b_conv2_t, b_conv2))
    sess.run(tf.assign(W_fc1_t, W_fc1))
    sess.run(tf.assign(b_fc1_t, b_fc1))
    sess.run(tf.assign(W_fc2_t, W_fc2))
    sess.run(tf.assign(b_fc2_t, b_fc2))

replay_memory_size = 1000
n_episodes = 2000
init_epsilon = 1.
final_epsilon = 0.1
final_exploration_episode = 500
target_update_frequency = 100
max_steps = 200
gamma = 0.99
minibatch_size = 32
# replay memory for s, a, r, terminal, and sn
Ds = np.zeros([replay_memory_size, env.ny, env.nx, env.nf], dtype=np.float32)
Da = np.zeros([replay_memory_size, env.na], dtype=np.float32)
Dr = np.zeros([replay_memory_size, 1], dtype=np.float32)
Dt = np.zeros([replay_memory_size, 1], dtype=np.float32)    # 1 if terminal
Dsn = np.zeros([replay_memory_size, env.ny, env.nx, env.nf], dtype=np.float32)
d = 0     # counter for storing in D
ds = 0    # number of valid entries in D
sum_rewards = np.zeros(n_episodes)

sess.run(tf.global_variables_initializer())
# specify which variables to save in *.ckpt and set max_to_keep = 0 to remove restriction on the # of files saved in one session
saver = tf.train.Saver({W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2}, max_to_keep = 0)

epsilon = init_epsilon
for episode in range(n_episodes):
    if episode % target_update_frequency == 0:
        update_target()
    s = env.reset()
    for t in range(max_steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(env.na)      # random action
        else:
            q = sess.run(y_hat, {x: np.reshape(s, [1, env.ny, env.nx, env.nf])})
            a = np.random.choice(np.where(q[0]==np.max(q))[0])          # greedy action with random tie break
        sn, r, terminal, _, _, _, _, _, _, _, _ = env.run(a - 1)        # action to take is -1, 0, 1
        sum_rewards[episode] += r
        Ds[d] = s
        Dr[d] = r
        Dt[d] = terminal
        Dsn[d] = sn
        Da[d] = 0; Da[d,a] = 1     # since Da[d,:] is a one-hot vector
        d = (d + 1) % replay_memory_size      # since D is a circular buffer
        ds = min(ds + 1, replay_memory_size)
        if ds == replay_memory_size:    # starts training once D is full
            c = np.random.choice(replay_memory_size, minibatch_size)
            x_train = Ds[c]
            qn = sess.run(y_hat_t, {x: Dsn[c]})
            y_train = Dr[c] + (1 - Dt[c]) * gamma * np.max(qn, axis=1, keepdims=True)
            a_train = Da[c]
            train_step.run({x: x_train, y_: y_train, a_: a_train})
        if terminal:
            break
        s = sn
    epsilon = max(final_epsilon, epsilon - 1. / final_exploration_episode) 
    if episode % 100 == 99:    # for saving *.ckpt and testing
        saver.save(sess, './breakout/breakout.ckpt', global_step = episode)  # saves *.ckpt every 100 episodes
        test_score = 0.
        s = env.reset()
        for t in range(max_steps):     # for testing with epsilon = 0
            q = sess.run(y_hat, {x: np.reshape(s, [1, env.ny, env.nx, env.nf])})
            a = np.random.choice(np.where(q[0]==np.max(q))[0])               # greedy action with random tie break
            sn, r, terminal, _, _, _, _, _, _, _, _ = env.run(a - 1)         # action to take is -1, 0, 1
            test_score += r
            if terminal:
                break
            s = sn
        # print test score and # of time steps to achieve the score
        print('episode: %d, train score: %f, test score: %f, test time steps: %d' % (episode, sum_rewards[episode], test_score, t + 1))
saver.save(sess, './breakout.ckpt')    # saves the final version
