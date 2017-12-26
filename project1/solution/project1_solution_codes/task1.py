# Task 1, Project 1
# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST
# Written by Sae-Young Chung, Nov. 5, 2017

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

nh = 20

x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([2, nh], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[nh]))
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([nh, 1], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[1]))
h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
y_hat = tf.nn.sigmoid(h_fc2)

# Training and evaluation of accuracy
cross_entropy = tf.reduce_mean(tf.nn.softplus((1-2*y_)*h_fc2))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.cast(y_hat>0.5,tf.float32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def generate_data(n):
    y = np.random.randint(2,size=[n,1])
    angle = np.random.rand(n,1)*2.*np.pi
    radius = np.random.randn(n,1)+y*5.
    x = np.column_stack((radius*np.cos(angle),radius*np.sin(angle)))
    return x,y;

x_train, y_train = generate_data(100000)
x_val, y_val = generate_data(1000)
x_test, y_test = generate_data(1000)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

plt.figure(1)
xmax=8.
x1s,x1e=-xmax,xmax

w=sess.run(W_fc1)
b=sess.run(b_fc1)
for i in range(y_test.size):
    if y_test[i] == 0:
        plt.plot(x_test[i][0],x_test[i][1],'ro')
    else:
        plt.plot(x_test[i][0],x_test[i][1],'g*')
lines=[]
for i in range(nh):
    x2s=-(w[0][i]*x1s+b[i])/w[1][i]
    x2e=-(w[0][i]*x1e+b[i])/w[1][i]
    line,=plt.plot([x1s,x1e],[x2s,x2e],'b-')
    lines.append(line)
plt.grid(True)
plt.axis([-xmax,xmax,-xmax,xmax])
plt.axes().set_aspect('equal')
plt.savefig('task1a0.png',dpi=600,bbox_inches='tight')
#plt.show(block=False)

plt.figure(2)
n_grid=50
x_grid=np.zeros((n_grid**2,2))
for i in range(n_grid):
    for j in range(n_grid):
        x_grid[i*n_grid+j][0]=(i+0.5)/n_grid*2.*xmax-xmax
        x_grid[i*n_grid+j][1]=(j+0.5)/n_grid*2.*xmax-xmax
y_grid=sess.run(y_hat, feed_dict={x: x_grid})
grids=[]
for i in range(n_grid**2):
    if y_grid[i]<0.5:
        grid,=plt.plot(x_grid[i][0],x_grid[i][1],'ro')
    else:
        grid,=plt.plot(x_grid[i][0],x_grid[i][1],'g*')
    grids.append(grid)
plt.axis([-xmax,xmax,-xmax,xmax])
plt.axes().set_aspect('equal')
plt.savefig('task1b0.png',dpi=600,bbox_inches='tight')
#plt.show()

print("=========================")
print("|Batch\t|Train\t|Val\t|")
print("=========================")
for i in range(y_train.size // 100):
    train_step.run(feed_dict={x: x_train[i*100:(i+1)*100], y_: y_train[i*100:(i+1)*100]})
    train_accuracy = accuracy.eval(feed_dict={x:x_train[i*100:(i+1)*100], y_: y_train[i*100:(i+1)*100]})
    val_accuracy = accuracy.eval(feed_dict={x: x_val, y_: y_val})
    print("|%d\t|%.4f\t|%.4f\t|"%(i+1, train_accuracy, val_accuracy))
    if i==9 or i==24 or i==49 or i==99 or i==999:
        plt.figure(1)
        w=sess.run(W_fc1)
        b=sess.run(b_fc1)
        for j in range(nh):
            x2s=-(w[0][j]*x1s+b[j])/w[1][j]
            x2e=-(w[0][j]*x1e+b[j])/w[1][j]
            lines[j].set_ydata([x2s,x2e])
        plt.savefig('task1a'+str(i+1)+'.png',dpi=600,bbox_inches='tight')
        plt.figure(2)
        y_grid=sess.run(y_hat, feed_dict={x: x_grid})
        for j in range(n_grid**2):
            if y_grid[j]<0.5:
                grids[j].set_marker('o')
                grids[j].set_color('r')
            else:
                grids[j].set_marker('*')
                grids[j].set_color('g')
        plt.savefig('task1b'+str(i+1)+'.png',dpi=600,bbox_inches='tight')
print("=========================")
test_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test})
print("Test accuracy: %.4f"%test_accuracy)

print(sess.run(W_fc1))
print(sess.run(b_fc1))
print(sess.run(W_fc2))
print(sess.run(b_fc2))

