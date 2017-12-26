# Task 3, Project 1
# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST
# Written by Sae-Young Chung, Nov. 5, 2017

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

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

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
y_hat = tf.nn.softmax(h_fc2)

# Training and evaluation of accuracy
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(y_hat),1))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_fc2))     # do this if TensorFlow ver. >= 1.0
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
print("=================================")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(5):
    for i in range(550):
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

# For Task 3
Psi=h_fc1.eval(feed_dict={x: mnist.test.images})
Psi_mean=np.mean(Psi,axis=0)
Phi=Psi-Psi_mean
W,S,V=np.linalg.svd(np.matmul(Phi.transpose(), Phi))
Z=np.matmul(Phi, W)

plt.figure(1)
plt.plot(Z[:,0],Z[:,1],'.')
#plt.savefig('task3a.png',dpi=600,bbox_inches='tight')
plt.show(block=False)

plt.figure(2)
nc=20
counter=np.zeros(nc,dtype=int)
# 10*nc examples containing 10 labels each
images100=np.zeros([10*nc,784],dtype=np.float32)
for i in range(mnist.test.num_examples):
    label = np.nonzero(mnist.test.labels[i])[0][0]
    if counter[label] < nc:
        images100[label*nc+counter[label]]=mnist.test.images[i]
        counter[label]+=1

Omega=h_fc1.eval(feed_dict={x: images100})
Q=np.matmul(Omega-Psi_mean,W)
colors=['red','green','blue','cyan','magenta','yellow','black','brown','indigo','lavender']
for i in range(10*nc):
    if i%nc == 0:
        cx,cy=0.,0.
    plt.plot(Q[i][0], Q[i][1], color=colors[i // nc], marker='o')
    cx+=Q[i][0];cy+=Q[i][1]
    if i%nc == nc-1:
        cx/=nc;cy/=nc
        plt.text(cx,cy,str(i//nc),va='center',ha='center',size=10,color=colors[i // nc],backgroundcolor='grey')
#plt.savefig('task3b.png',dpi=600,bbox_inches='tight')
plt.show()

