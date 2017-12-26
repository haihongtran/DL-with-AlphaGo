# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/19/2017

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from breakout_env import *
from wait import *


class breakout_animation(animation.TimedAnimation):
    def __init__(self, sess, env, max_steps, ckpt_file = '', frames_per_step = 5):
        self.sess = sess
        self.env = env
        self.max_steps = max_steps

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        self.objs = []

        # boundary
        w = 0.1
        ax.plot([-w,-w,env.nx+w,env.nx+w],[0,env.ny+w,env.ny+w,0],'k-',linewidth=5)

        # bricks
        wb = 0.05
        self.bricks = []
        self.brick_colors = [['red'], ['blue','red'], ['blue','green','red'], ['blue','green','yellow','red'], ['blue','green','yellow','orange','red'], \
            ['purple','blue','green','yellow','brown','orange','red'], ['purple','blue','green','yellow','brown','orange','red']]    # add more colors if needed
        for y in range(self.env.nb):
            b = []
            yp = y + (self.env.ny - self.env.nt - self.env.nb)
            for x in range(self.env.nx):
                b.append(patches.Rectangle((x + wb, yp + wb), 1-2*wb, 1-2*wb, edgecolor='none', facecolor=self.brick_colors[self.env.nb-1][y]))
                ax.add_patch(b[x])
                self.objs.append(b[x])
            self.bricks.append(b)
 
        # ball
        self.ball = patches.Circle(env.get_ball_pos(0.), radius = 0.15, color = 'red')
        ax.add_patch(self.ball)
        self.objs.append(self.ball)

        # score text
        self.text = ax.text(0.5 * env.nx, 0, '', ha='center')
        self.objs.append(self.text)

        # game over text
        self.gameover_text = ax.text(0.5 * env.nx, 0.5 * env.ny, '', ha='center')
        self.objs.append(self.gameover_text)

        self.frames_per_step = frames_per_step
        self.total_frames = self.frames_per_step * self.max_steps

        # paddle
        self.paddle = patches.Rectangle((env.p, 0.5), 1, 0.5, edgecolor='none', facecolor='red')
        ax.add_patch(self.paddle)

        # for early termination of animation
        self.iter_objs = []
        self.iter_obj_cnt = 0

        # interval = 50msec
        animation.TimedAnimation.__init__(self, fig, interval=50, repeat=False, blit=False)

    def _draw_frame(self, k):
        if self.terminal:
            return
        if k == 0:
            self.iter_obj_cnt -= 1
        if k % self.frames_per_step == 0:
            q = self.sess.run(y_hat, {x: np.reshape(self.s, [1, env.ny, env.nx, env.nf])})
            self.a = np.random.choice(np.where(q[0]==np.max(q))[0]) - 1         # greedy action with random tie break
            self.p = self.env.p
            self.pn = min(max(self.p + self.a, 0), self.env.nx - 1)

        t = (k % self.frames_per_step) * 1. / self.frames_per_step
        self.ball.center = self.env.get_ball_pos(t)
        self.paddle.set_x(t * self.pn + (1-t) * self.p)

        if k % self.frames_per_step == self.frames_per_step - 1:
            sn, reward, terminal, p0, p, bx0, by0, vx0, vy0, rx, ry = self.env.run(self.a)
            self.sum_reward += reward
            if reward > 0.:
                self.bricks[ry][rx].set_facecolor('none')
                self.text.set_text('Score: %d' % self.sum_reward)
            if terminal:
                self.terminal = terminal
                self.gameover_text.set_text('Game Over')
                for _ in range(self.total_frames - k - 1):
                    self.iter_objs[self.iter_obj_cnt].next()     # for early termination of animation (latest iterator is used first)
            self.s = sn

        self._drawn_artists = self.objs

    def new_frame_seq(self):
        iter_obj = iter(range(self.total_frames))
        self.iter_objs.append(iter_obj)
        self.iter_obj_cnt += 1
        return iter_obj

    def _init_draw(self):
        self.s = self.env.reset()    # self.s added for neural net version
        self.sum_reward = 0.
        self.p = self.env.p    # current paddle position
        self.pn = self.p       # next paddle position
        self.a = 0             # action
        self.terminal = 0

        for y in range(self.env.nb):
            for x in range(self.env.nx):
                self.bricks[y][x].set_facecolor(self.brick_colors[self.env.nb-1][y])

        self.ball.center = self.env.get_ball_pos(0.)
        self.paddle.set_x(self.p)

        self.text.set_text('Score: 0')
        self.gameover_text.set_text('')


env = breakout_environment(5, 8, 3, 1, 2)

c1 = 30
c2 = 30
f1 = 128

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, env.ny, env.nx, env.nf])

# First convolutional layer
W_conv1 = tf.Variable(tf.constant(0., shape = [3, 3, env.nf, c1]))
b_conv1 = tf.Variable(tf.constant(0., shape = [c1]))
h_conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_relu1 = tf.nn.relu(h_conv1 + b_conv1)

# Second convolutional layer
W_conv2 = tf.Variable(tf.constant(0., shape = [3, 3, c1, c2]))
b_conv2 = tf.Variable(tf.constant(0., shape = [c2]))
h_conv2 = tf.nn.conv2d(h_relu1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_relu2 = tf.nn.relu(h_conv2 + b_conv2)

# Fully-connected Layer
W_fc1 = tf.Variable(tf.constant(0., shape = [env.ny * env.nx * c2, f1]))
b_fc1 = tf.Variable(tf.constant(0., shape = [f1]))
h_flat = tf.reshape(h_relu2, [-1, env.ny * env.nx * c2])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

# Output layer
W_fc2 = tf.Variable(tf.constant(0., shape = [f1, env.na]))
b_fc2 = tf.Variable(tf.constant(0., shape = [env.na]))
y_hat = tf.matmul(h_fc1, W_fc2) + b_fc2

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, './breakout.ckpt')

ani_max_steps = 200
ani = breakout_animation(sess, env, ani_max_steps)
# uncomment the following to save animation as *.mp4 (don't do plt.show() in that case since otherwise animation will run one more time with possibly different outcome)
#ani.save('breakout_sol.mp4', dpi=200)
plt.show(block=False)
wait('Press enter to quit')

