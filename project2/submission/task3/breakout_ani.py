# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Project 2 - Task 3
# Tran Hong Hai

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from breakout_env import *
from wait import *

env = breakout_environment(5, 8, 3, 1, 2)

def q_network(X_input, env, name):
    with tf.variable_scope(name) as scope:
        conv1 = tf.layers.conv2d(X_input, filters = 20, kernel_size = [3,3],\
            strides = 1, padding = 'VALID', activation = tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, filters = 40, kernel_size = [2,2],\
            strides = 1, padding = 'VALID', activation = tf.nn.relu)
        conv2_flat = tf.reshape(conv2, shape = [-1, 5*2*40])
        fc1 = tf.layers.dense(conv2_flat, units = 100, activation = tf.nn.relu)
        outputs = tf.layers.dense(fc1, env.na)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,\
                                        scope = scope.name)
    trainable_vars_name = {var.name[len(scope.name):]: var\
                            for var in trainable_vars}
    return outputs, trainable_vars_name

X_input = tf.placeholder(tf.float32, shape = [None, env.ny, env.nx, env.nf])
train_q_vals , train_vars  = q_network(X_input, env, name = "q_network/train")

# Load trained parameters of network
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

try:
    saver.restore(sess, "./breakout.ckpt")
    print 'Loading trained model'
except:
    print 'A new model is being trained'

class breakout_animation(animation.TimedAnimation):
    def __init__(self, env, max_steps, frames_per_step = 5):
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
            q_values = train_q_vals.eval(feed_dict = {X_input: np.reshape(self.env.s, (1, 8, 5, 2))})
            self.a = np.argmax(q_values) - 1    # Optimal action
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
        self._drawn_artists = self.objs

    def new_frame_seq(self):
        iter_obj = iter(range(self.total_frames))
        self.iter_objs.append(iter_obj)
        self.iter_obj_cnt += 1
        return iter_obj

    def _init_draw(self):
        _ = self.env.reset()
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

ani = breakout_animation(env, 200)
ani.save('breakout.mp4', dpi=200)
# plt.show(block=False)
# wait('Press enter to quit')
