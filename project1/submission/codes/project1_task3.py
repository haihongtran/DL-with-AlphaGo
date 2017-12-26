# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST
# Author: Tran Hong Hai <haitranhong@kaist.ac.kr>
# Project 1 - Task 3
# TensorFlow version: 1.3.0 - NumPy version: 1.13.3 - MatPlotLib version: 1.5.1

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x_ = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Convolutional layer
x_image = tf.reshape(x_, [-1,28,28,1])
W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 30], stddev=0.1))
b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv)
h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([12 * 12 * 30, 500], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
h_pool_flat = tf.reshape(h_pool, [-1, 12*12*30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_hat=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Train and Evaluate the Model
cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
print '================================='
print '|Epoch\tBatch\t|Train\t|Val\t|'
print '|===============================|'
for j in range(5):
    for i in range(550):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x_: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x_:batch[0], y_: batch[1]})
            val_accuracy = accuracy.eval(feed_dict=\
                {x_: mnist.validation.images, y_:mnist.validation.labels})
            print '|%d\t|%d\t|%.4f\t|%.4f\t|'%(j+1, i+1, train_accuracy, val_accuracy)
print '|===============================|'
test_accuracy = accuracy.eval(feed_dict=\
    {x_: mnist.test.images, y_:mnist.test.labels})
print 'test accuracy=%.4f'%(test_accuracy)

vectorNum = 10000
mat1 = h_fc1.eval(feed_dict = { x_: mnist.test.images[0:vectorNum, :]})
mat2 = mat1 - np.mean(mat1, axis = 0)
W, s, V = np.linalg.svd(np.dot(mat2.T, mat2))
Z = np.dot(mat2, W)

# Figure 1: Plot first 2 columns of Z
c = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'yellowgreen', 'orange', 'brown']
y_val = mnist.test.labels[0:vectorNum, :]
plt.figure(1)
ax = plt.subplot(111)
for currDigit in range(10):
    a = np.zeros((10))
    a[currDigit] = 1
    b = np.where(np.all(y_val == a, axis = 1))
    x0 = Z[b,0]
    x1 = Z[b,1]
    ax.scatter(x0, x1, s = 50, c = c[currDigit], label = currDigit, linewidths = 0.)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()

# Get 100 examples from original data set
dataSetImages = []
dataSetLabels = []
dataSetSize = 100
expEachDigit = 10
digitNum = dataSetSize/expEachDigit

for currDigit in range(digitNum):
    i = j = 0
    while ( (i < expEachDigit) and (j < mnist.test.labels.shape[0]) ):
        argMax = np.argmax(mnist.test.labels[j,:])
        if ( argMax == currDigit ):
            dataSetImages.append(mnist.test.images[j,:])
            dataSetLabels.append(mnist.test.labels[j,:])
            i += 1
        j += 1

dataSetImages = np.array(dataSetImages)
dataSetLabels = np.array(dataSetLabels)

omega = h_fc1.eval(feed_dict = {x_: dataSetImages, y_: dataSetLabels})
Q = np.dot(omega, W)

# Figure 2: Plot first two columns of Q
plt.figure(2)
ax = plt.subplot(111)
for i in range(10):
    plt.scatter(Q[(i*10):((i+1)*10),0], Q[(i*10):((i+1)*10),1],\
            s = 50, c = c[i], label = i)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()
