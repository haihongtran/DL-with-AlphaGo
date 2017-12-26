# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST
# Author: Tran Hong Hai <haitranhong@kaist.ac.kr>
# Project 1 - Task 2
# TensorFlow version: 1.3.0 - NumPy version: 1.13.3 - MatPlotLib version: 1.5.1

import tensorflow as tf
import numpy as np

# Get MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# Parameters
epochNum = 20
lr = 1e-4

# Remove number '9' from training set
filterTrainImages = []
filterTrainLabels = []
for i in range(mnist.train.labels.shape[0]):
    if ( np.argmax(mnist.train.labels[i,:]) != 9 ):
        filterTrainImages.append(mnist.train.images[i,:])
        filterTrainLabels.append(mnist.train.labels[i,0:9])
filterTrainImages = np.array(filterTrainImages)
filterTrainLabels = np.array(filterTrainLabels)

# Remove number '9' from validation set
filterValidImages = []
filterValidLabels = []
for i in range(mnist.validation.labels.shape[0]):
    if ( np.argmax(mnist.validation.labels[i,:]) != 9 ):
        filterValidImages.append(mnist.validation.images[i,:])
        filterValidLabels.append(mnist.validation.labels[i,0:9])
filterValidImages = np.array(filterValidImages)
filterValidLabels = np.array(filterValidLabels)

# Remove number '9' from test set
filterTestImages = []
filterTestLabels = []
for i in range(mnist.test.labels.shape[0]):
    if ( np.argmax(mnist.test.labels[i,:]) != 9 ):
        filterTestImages.append(mnist.test.images[i,:])
        filterTestLabels.append(mnist.test.labels[i,0:9])
filterTestImages = np.array(filterTestImages)
filterTestLabels = np.array(filterTestLabels)

# Tensorflow session
sess = tf.InteractiveSession()

# Place to hold data
x_ = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 9])

# First convolutional layer
x_image = tf.reshape(x_, [-1,28,28,1])
W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 30], stddev=0.1))
b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv)
h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Second convolutional layer
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 30, 50], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[50]))
h_conv2 = tf.nn.conv2d(h_pool, W_conv2, strides=[1, 1, 1, 1], padding='VALID')
h_relu2 = tf.nn.relu(h_conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([5 * 5 * 50, 500], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
h_pool_flat = tf.reshape(h_pool2, [-1, 5*5*50])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([500, 9], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[9]))
y_hat = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and Evaluate the Model
crossEntropy = - tf.reduce_sum(y_*tf.log(y_hat))
optimizer = tf.train.AdamOptimizer(lr)
training = optimizer.minimize(crossEntropy)
correctPred = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# Transfer Learning: Place holder for labels
y_tl_ = tf.placeholder(tf.float32, shape=[None, 10])
# Transfer Learning: New output layer
W_fc2_tl = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc2_tl = tf.Variable(tf.constant(0.1, shape=[10]))
y_hat_tl = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2_tl) + b_fc2_tl)
# Transfer Learning: Train and Evaluate the Model with new output layer
crossEntropyTL = - tf.reduce_sum(y_tl_*tf.log(y_hat_tl))
trainingTL = optimizer.minimize(crossEntropyTL, var_list = (W_fc2_tl, b_fc2_tl))
correctPredTL = tf.equal(tf.argmax(y_hat_tl, 1), tf.argmax(y_tl_, 1))
accuracyTL = tf.reduce_mean(tf.cast(correctPredTL, tf.float32))

# Initialize all declared tensors
tf.global_variables_initializer().run()

# Training model
trainingSetSize = filterTrainImages.shape[0]
miniBatchSize = 100
miniBatchNum = int(np.ceil(trainingSetSize/miniBatchSize))
print '================================='
print '|Epoch\t|MnBatch|Train\t|Val\t|'
print '|===============================|'
for j in range(epochNum):
    for i in range(miniBatchNum):
        # Extract mini batch data
        if ( i < miniBatchNum - 1 ):
            trainingImages = filterTrainImages[(miniBatchSize*i) : (miniBatchSize*(i+1)), :]
            trainingLabels = filterTrainLabels[(miniBatchSize*i) : (miniBatchSize*(i+1)), :]
        else:   # i = miniBatchNum - 1
            restTrainingExps = trainingSetSize - miniBatchSize*i
            trainingImages = filterTrainImages[(miniBatchSize*i) : (miniBatchSize*i + restTrainingExps), :]
            trainingLabels = filterTrainLabels[(miniBatchSize*i) : (miniBatchSize*i + restTrainingExps), :]
        # Start training
        training.run(feed_dict={x_: trainingImages, y_: trainingLabels, keep_prob: 0.5})
        # Print train and validation accuracy during training
        if ((i%100 == 99) or (i == miniBatchNum - 1)):
            trainAccuracy = accuracy.eval(feed_dict={x_:trainingImages, y_: trainingLabels, keep_prob: 1.})
            valAccuracy = accuracy.eval(feed_dict=\
                {x_: filterValidImages, y_: filterValidLabels, keep_prob: 1.})
            print '|%d\t|%d\t|%.4f\t|%.4f\t|' % (j+1, i+1, trainAccuracy, valAccuracy)
print '|===============================|'

# Test accuracy of pre-trained model
testAccuracy = accuracy.eval(feed_dict=\
    {x_: filterTestImages, y_: filterTestLabels, keep_prob: 1.})
print 'Test accuracy of pre-trained model (9 outputs) is %.4f' % (testAccuracy)

# Transfer Learning
print '================================='
print '|Epoch\t|MnBatch|Train\t|Val\t|'
print '|===============================|'
for j in range(epochNum):
    for i in range(550):
        # Extract mini batch data
        batch = mnist.train.next_batch(100)
        # Start training
        trainingTL.run(feed_dict={x_: batch[0], y_tl_: batch[1], keep_prob: 0.5})
        # Print training and validation accuracy during training
        if (i%100 == 99) or (i == 549):
            trainAccuracy = accuracyTL.eval(feed_dict={x_:batch[0], y_tl_: batch[1], keep_prob: 1.})
            valAccuracy = accuracyTL.eval(feed_dict=\
                {x_: mnist.validation.images, y_tl_:mnist.validation.labels, keep_prob: 1.})
            print '|%d\t|%d\t|%.4f\t|%.4f\t|' % (j+1, i+1, trainAccuracy, valAccuracy)
print '|===============================|'

# Test accuracy of transfer learning model
testAccuracyNew = accuracyTL.eval(feed_dict=\
    {x_: mnist.test.images, y_tl_:mnist.test.labels, keep_prob: 1.})
print 'Test accuracy of transfer learning model (10 outputs) is %.4f' % (testAccuracyNew)
