# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Project 2 - Task 3
# Tran Hong Hai

import numpy as np
import tensorflow as tf
import random
from breakout_env import *

# An instance of environment
env = breakout_environment(5, 8, 3, 1, 2)

xavier_init = tf.contrib.layers.xavier_initializer(uniform = False)
xavier_conv_init = tf.contrib.layers.xavier_initializer_conv2d(uniform = False)
def q_network(X_input, env, name):
    with tf.variable_scope(name) as scope:
        conv1 = tf.layers.conv2d(X_input, filters = 20, kernel_size = [3,3],\
            strides = 1, padding = 'VALID', activation = tf.nn.relu,\
	    kernel_initializer = xavier_conv_init)
        conv2 = tf.layers.conv2d(conv1, filters = 40, kernel_size = [2,2],\
            strides = 1, padding = 'VALID', activation = tf.nn.relu,\
	    kernel_initializer = xavier_conv_init)
        conv2_flat = tf.reshape(conv2, shape = [-1, 5*2*40])
        fc1 = tf.layers.dense(conv2_flat, units = 100, activation = tf.nn.relu,\
	    kernel_initializer = xavier_init)
        outputs = tf.layers.dense(fc1, env.na, kernel_initializer = xavier_init)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,\
                                        scope = scope.name)
    trainable_vars_name = {var.name[len(scope.name):]: var\
                            for var in trainable_vars}
    return outputs, trainable_vars_name

# Define networks
X_input = tf.placeholder(tf.float32, shape = [None, env.ny, env.nx, env.nf])
train_q_vals , train_vars  = q_network(X_input, env, name = "q_network/train")
target_q_vals, target_vars = q_network(X_input, env, name = "q_network/target")

# Copy train DQN to target DQN
copy_ops = [target_var.assign(train_vars[var_name])
            for var_name, target_var in target_vars.items()]
update_target_dqn = tf.group(*copy_ops)

# Define training configs
learning_rate = 1e-4
with tf.variable_scope("training"):
    X_action = tf.placeholder(tf.int32, shape = [None])
    target_q = tf.placeholder(tf.float32, shape = [None, 1])
    X_act_onehot = tf.one_hot(X_action, env.na, dtype = tf.float32)
    train_q = tf.reduce_sum(tf.multiply(train_q_vals, X_act_onehot),\
                            axis = 1, keep_dims = True)
    cost = tf.reduce_sum(tf.square(target_q - train_q))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training = optimizer.minimize(cost)

# Class ExperienceMemory to store experience
class ExperienceMemory():
    def __init__(self, memorySize = 1000):
        self.memory = []
        self.memorySize = memorySize
    def add(self, experience):
        if len(self.memory) + len(experience) >= self.memorySize:
            self.memory[0:(len(experience)+len(self.memory))-self.memorySize] = []
        self.memory.extend(experience)
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.memory, size)),[size, 5])

# Start tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

# Training parameters
gamma = 0.99            # Discount rate
n_episodes = 2000       # Number of episodes
max_steps = 200         # Max steps in one episode
memorySize = 5000       # Size of experience memory
preTrainSteps = 5000    # Step of random actions before training begins
miniBatchSize = 32      # Experience taken from memory in each training iteration
totalSteps = 0          # Used to track when to start training
targetUpdateStep = 500  # Number of training steps to wait to update target network
trainingStep = 0        # Used to track the time to update target network
initEpsilon = 1.        # Initial value of epsilon
endEpsilon = 0.         # End value of epsilon
epsilonStep = 1./1000   # Epsilon changing rate
epsilon = initEpsilon   # Epsilon value used for e-greedy

# Init experience memory
expMem = ExperienceMemory(memorySize)

# Copy initial value of weights from trainQN to target QN
update_target_dqn.run()

# DQN with experience replay
for episode in range(n_episodes):
    s = env.reset()
    reward = 0
    for step in range(max_steps):
        if np.random.rand() < epsilon or totalSteps < preTrainSteps:
            a = np.random.randint(env.na)   # random action
        else:   # Optimal action
            y_hat = train_q_vals.eval(feed_dict = {X_input: np.reshape(s, [1, env.ny, env.nx, env.nf])})
            a = np.random.choice(np.where(y_hat[0]==np.max(y_hat))[0])
        # Next step in environment
        sn, r, terminal, _, _, _, _, _, _, _, _ = env.run(a - 1)
        totalSteps += 1
        reward += r
        # Add experience to memory
        expMem.add(np.array([s, a, r, sn, terminal]).reshape(1,5))
        # Check if training can be started
        if totalSteps > preTrainSteps:
            # Sample random minibatch
            miniBatch = expMem.sample(miniBatchSize)
            sBatch   = np.vstack(miniBatch[:,0]).reshape(miniBatchSize, env.ny, env.nx, env.nf)
            aBatch   = np.vstack(miniBatch[:,1]).reshape(miniBatchSize)
            # Calculate target Q value
            q_target = []
            for i in range(miniBatchSize):
                if miniBatch[i,4] == 1:     # If terminal state
                    q_target.append(miniBatch[i,2])
                else:
                    y_target = target_q_vals.eval(\
                        feed_dict = {X_input: np.reshape(miniBatch[i,3], [1, env.ny, env.nx, env.nf])})
                    q_target.append(miniBatch[i,2] + gamma * np.max(y_target))
            q_target = np.array(q_target).reshape(miniBatchSize,1)
            # Start training
            training.run(feed_dict = {X_input: sBatch, X_action: aBatch, target_q: q_target})
            trainingStep += 1
            # Update target network
            if trainingStep % targetUpdateStep == 0:
                update_target_dqn.run()
        if terminal:
            if totalSteps > preTrainSteps and epsilon > endEpsilon:
                epsilon -= epsilonStep
            if totalSteps > preTrainSteps:
                currCost = cost.eval(feed_dict = {X_input: sBatch, X_action: aBatch, target_q: q_target})
                print 'Episode: %d\tStep: %3d\tReward: %2d\tCost: %f\tEpsilon: %f'\
                    %(episode+1, step+1, reward, currCost, epsilon)
            break
        # Update two most recent frames
        s = sn

# Save trained parameters
save_path = saver.save(sess, "./breakout.ckpt")
