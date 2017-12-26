# Task 2, Project 1
# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST
# Written by Sae-Young Chung, Nov. 5, 2017

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ is for the second network with 10 outputs
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# y9 is playing the role of y_ for the first network with 9 outputs
y9 = tf.placeholder(tf.float32, shape=[None, 9])

# exclude all examples with label==label_to_exclude
def exclude_label(original_images, original_labels, label_to_exclude):
    n=original_images.shape[0]
    images=np.zeros([n,original_images.shape[1]],dtype=np.float32)
    labels=np.zeros([n,original_labels.shape[1]-1],dtype=np.float32)
    m=0
    for i in range(n):
        if original_labels[i][label_to_exclude]==0.:
            images[m]=original_images[i]
            labels[m]=np.concatenate((original_labels[i][:label_to_exclude],original_labels[i][label_to_exclude+1:]))
            m+=1
    images=np.resize(images,(m,images.shape[1]))
    labels=np.resize(labels,(m,labels.shape[1]))
    return m, images, labels

# remove examples with label '7' and generate new datasets
n_train_9, train_images_9, train_labels_9 = exclude_label(mnist.train.images, mnist.train.labels, 7)
n_val_9, val_images_9, val_labels_9 = exclude_label(mnist.validation.images, mnist.validation.labels, 7)
n_test_9, test_images_9, test_labels_9 = exclude_label(mnist.test.images, mnist.test.labels, 7)

# random permutation for the training set with 9 labels
# (note that mnist.train.next_batch() will perform random permutation for mnist.train dataset)
perm = np.arange(n_train_9)
np.random.shuffle(perm)
train_images_9 = train_images_9[perm]
train_labels_9 = train_labels_9[perm]

# Convolutional layer
x_image = tf.reshape(x, [-1,28,28,1])
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

# Output layer with 9 outputs
W_fc2_9 = tf.Variable(tf.truncated_normal([500, 9], stddev=0.1))
b_fc2_9 = tf.Variable(tf.constant(0.1, shape=[9]))
h_fc2_9 = tf.matmul(h_fc1, W_fc2_9) + b_fc2_9
y_hat_9 = tf.nn.softmax(h_fc2_9)

# Output layer with 10 outputs
W_fc2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
y_hat = tf.nn.softmax(h_fc2)

# Training & evaluation for the network with 9 outputs 
# This will only train W_conv, b_conv, W_fc1, b_fc1, W_fc2_9, b_fc2_9
# W_fc2_9 and b_fc2_9 will not be trained although we didn't exclude them in optimization
#   since they do not affect the cost function 'cross_entropy_9'
cross_entropy_9 = -tf.reduce_mean(tf.reduce_sum(y9*tf.log(y_hat_9),1))
#cross_entropy_9 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y9, logits=h_fc2_9))   # do this if TensorFlow ver >= 1.0
train_step_9 = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_9)
correct_prediction_9 = tf.equal(tf.argmax(y_hat_9,1), tf.argmax(y9,1))
accuracy_9 = tf.reduce_mean(tf.cast(correct_prediction_9, tf.float32))

# Training & evaluation for the network with 10 outputs
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(y_hat),1))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_fc2))   # do this if TensorFlow ver >= 1.0
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy, var_list=[W_fc2,b_fc2]) # only train the last layer
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training the first network with 9 outputs
sess.run(tf.initialize_all_variables())
print("=================================")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(5):
    for i in range(n_train_9 // 100):
        train_step_9.run(feed_dict={x: train_images_9[i*100:i*100+100], y9: train_labels_9[i*100:i*100+100]})
        if i%50 == 49:
            train_accuracy = accuracy_9.eval(feed_dict={x: train_images_9[i*100:i*100+100], y9: train_labels_9[i*100:i*100+100]})
            val_accuracy = accuracy_9.eval(feed_dict=\
                {x: val_images_9, y9: val_labels_9})
            print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))
print("|===============================|")
test_accuracy = accuracy_9.eval(feed_dict=\
    {x: test_images_9, y9: test_labels_9})
print("test accuracy=%.4f"%(test_accuracy))

#print(sess.run(W_fc1))
#print(sess.run(b_fc1))
#print(sess.run(W_fc2))
#print(sess.run(b_fc2))

# Training the last layer of the second network with 10 outputs while keeping the parameters of the other layers unchanged
print("=================================")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(5):
    for i in range(mnist.train.num_examples // 100):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
            val_accuracy = accuracy.eval(feed_dict=\
                {x: mnist.validation.images, y_:mnist.validation.labels})
            print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))
print("|===============================|")
test_accuracy = accuracy.eval(feed_dict=\
    {x: mnist.test.images, y_:mnist.test.labels})
print("test accuracy=%.4f"%(test_accuracy))

#print(sess.run(W_fc1))
#print(sess.run(b_fc1))
#print(sess.run(W_fc2))
#print(sess.run(b_fc2))

