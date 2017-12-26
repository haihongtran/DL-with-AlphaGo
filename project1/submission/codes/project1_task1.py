# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST
# Author: Tran Hong Hai <haitranhong@kaist.ac.kr>
# Project 1 - Task 1
# TensorFlow version: 1.3.0 - NumPy version: 1.13.3 - MatPlotLib version: 1.5.1

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Neural network settings
hidNeus = 20
epochNum = 1000
lr = 0.0001

# Generating training set
trainingSetSize = 10000
x_train = np.zeros((trainingSetSize, 2), dtype = np.float32)
y_train = np.zeros((trainingSetSize, 1), dtype = np.float32)
r_train = np.random.normal(size = trainingSetSize)
t_train = np.random.uniform(0, 2*pi, trainingSetSize)
for i in range(trainingSetSize):
    y_train[i] = np.random.randint(2)   # 0 or 1
    if ( y_train[i] == 0 ):  # x = [r*cos(t), r*sin(t)]
        x_train[i,:] = [r_train[i] * np.cos(t_train[i]), r_train[i] * np.sin(t_train[i])]
    else:   # x = [(r+5)*cos(t), (r+5)*cos(t)]
        x_train[i,:] = [(r_train[i] + 5) * np.cos(t_train[i]), (r_train[i] + 5) * np.sin(t_train[i])]

# Generating validation set
validSetSize = 1000
x_valid = np.zeros((validSetSize, 2), dtype = np.float32)
y_valid = np.zeros((validSetSize, 1), dtype = np.float32)
r_valid = np.random.normal(size = validSetSize)
t_valid = np.random.uniform(0, 2*pi, validSetSize)
for i in range(validSetSize):
    y_valid[i] = np.random.randint(2)   # 0 or 1
    if ( y_valid[i] == 0 ):  # x = [r*cos(t), r*sin(t)]
        x_valid[i,:] = [r_valid[i] * np.cos(t_valid[i]), r_valid[i] * np.sin(t_valid[i])]
    else:   # x = [(r+5)*cos(t), (r+5)*cos(t)]
        x_valid[i,:] = [(r_valid[i] + 5) * np.cos(t_valid[i]), (r_valid[i] + 5) * np.sin(t_valid[i])]

# Generating test set
testSetSize = 1000
x_test = np.zeros((testSetSize, 2), dtype = np.float32)
y_test = np.zeros((testSetSize, 1), dtype = np.float32)
r_test = np.random.normal(size = testSetSize)
t_test = np.random.uniform(0, 2*pi, testSetSize)
for i in range(testSetSize):
    y_test[i] = np.random.randint(2)   # 0 or 1
    if ( y_test[i] == 0 ):  # x = [r*cos(t), r*sin(t)]
        x_test[i,:] = [r_test[i] * np.cos(t_test[i]), r_test[i] * np.sin(t_test[i])]
    else:   # x = [(r+5)*cos(t), (r+5)*cos(t)]
        x_test[i,:] = [(r_test[i] + 5) * np.cos(t_test[i]), (r_test[i] + 5) * np.sin(t_test[i])]

# Places to hold data
x_ = tf.placeholder(dtype = tf.float32, shape = [None, 2])
y_ = tf.placeholder(dtype = tf.float32, shape = [None, 1])

# Params initialization
W = tf.Variable(tf.truncated_normal([2, hidNeus], stddev = 0.1))
c = tf.Variable(tf.constant(0, shape = [hidNeus], dtype = tf.float32))
w = tf.Variable(tf.truncated_normal([hidNeus, 1], stddev = 0.1))
b = tf.Variable(tf.constant(0, shape = [1], dtype = tf.float32))

# Hidden layer implementation
h_relu = tf.nn.relu(tf.matmul(x_, W) + c)

# Output layer implementation
z = tf.matmul(h_relu, w) + b
y_hat = tf.nn.sigmoid(z)

# Cost function
cost = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_, logits = z)

# Training settings
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(cost)

# Evaluation of the model
y_pred = tf.round(y_hat)
correct_pred = tf.equal(y_pred, y_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Learning
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

miniBatchSize = 1000
miniBatchNum = trainingSetSize/miniBatchSize
print "========================="
print "|Epoch\t|Train\t|Val\t|"
print "|=======================|"
for i in range(epochNum):
    # Extract mini batch data and start training
    for j in range(miniBatchNum):
        trainingData = x_train[(miniBatchSize*j) : (miniBatchSize*(j+1)), :]
        trainingLabels = y_train[(miniBatchSize*j) : (miniBatchSize*(j+1)), :]
        train.run(feed_dict={x_: trainingData, y_: trainingLabels})
    # Print train and validation accuracy during training
    if i%10 == 9:
        trainAccuracy = accuracy.eval(feed_dict = {x_:trainingData, y_: trainingLabels})
        validAccuracy = accuracy.eval(feed_dict = {x_: x_valid, y_: y_valid})
        print "|%d\t|%.4f\t|%.4f\t|" %(i+1, trainAccuracy, validAccuracy)
print '|=======================|'

testAccuracy = accuracy.eval(feed_dict = {x_: x_test, y_: y_test})
print "Test accuracy is %.4f" %(testAccuracy)

print 'W =', W.eval()
print 'c =', c.eval()
print 'w =', w.eval()
print 'b =', b.eval()

plt.figure(1)
# Plot test data
for i in range(testSetSize):
    if ( y_test[i] == 0 ):
        plt.scatter(x_test[i,0], x_test[i,1], s = 50, c = 'b', marker = 'o')
    else:
        plt.scatter(x_test[i,0], x_test[i,1], s = 50, c = 'g', marker = 'v')

W_val = W.eval()
c_val = c.eval()

# Plot line dividing active and inactive region for hidden ReLU layer
for i in range(hidNeus):
    a = W_val[0,i]; b = W_val[1,i]; c = c_val[i]
    x_plot = np.arange(-10, 11)
    plt.plot(x_plot, (-a*x_plot-c)/b, linewidth = 1, color = 'r')

xmin=-10.
xmax= 10.
ymin=-10.
ymax= 10.
plt.axis([xmin,xmax,ymin,ymax])
plt.grid(True)
plt.show()
