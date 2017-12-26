# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
#
# Written by Sae-Young Chung, 2017/12/21
#   for EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017


import tensorflow as tf
import numpy as np
from boardgame import game1, game2, game3, game4, data_augmentation
import sys

class tictactoe_with_arbitrary_players(game2):
    def next_move(self, b, state, game_in_progress, net, rn, p, move, nlevels = 1, rw = 0):
        # this overrides the original 'next_move' function
        # rn = 0,1,2,... is the index for player's behavior if net==[]
        # rn is from 0 to (9*7*5*3 - 1) for black and is from 0 to (8*6*4*2 - 1) for white
        # assume nlevels == 1

        # if neural network is used, call the original next_move function
        if net != []:
            return super(tictactoe_with_arbitrary_players, self).next_move(b, state, game_in_progress, net, rn, p, move, nlevels, rw)

        # board size
        nx = self.nx; ny = self.ny; nxy = nx * ny
        # number of games
        if b.ndim>=3:
            ng = b.shape[2]
        else:
            ng=1
        # number of valid moves in each game
        n_valid_moves = np.zeros((ng))
        # check whether each of up to 'nxy' moves is valid for each game
        valid_moves = np.zeros((ng, nxy))
        # win probability for each next move (does not play any role if net == [])
        wp_all = np.zeros((nx, ny, ng))
        # maximum of wp_all over all possible next moves (does not play any role if net == [])
        wp_max = -np.ones((ng))
        x = -np.ones((ng))
        y = -np.ones((ng))

        for p1 in range(nxy):
            vm1, b1, state1 = self.valid(b, state, self.xy(p1), p)
            n_valid_moves += vm1
            valid_moves[:, p1] = vm1

        # n_valid_moves is 0 if game is not in progress
        n_valid_moves = n_valid_moves * game_in_progress

        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b[:, :, :]
        new_state = np.zeros(state.shape)
        new_state[:, :] = state[:, :]

        for k in range(ng):
            if n_valid_moves[k]: # if there are valid moves
                next_behavior = rn[k]
                for l in range(int(n_valid_moves[k]) + 2, nxy + 1, 2):
                    next_behavior //= l
                next_behavior %= n_valid_moves[k]

                for i in range(nx):
                    for j in range(ny):
                        if b[i, j, k] == 0:
                            if next_behavior == 0:
                                rxy = np.array([[i, j]])
                                isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], rxy, p)
                                new_board[:, :, [k]] = bn
                                new_state[:, [k]] = sn
                                x[k] = i
                                y[k] = j
                            next_behavior -= 1
            else: # if there is no more valid move
                isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], -np.ones((1, 2)), p)
                new_state[:, [k]] = sn

        return new_board, new_state, n_valid_moves, wp_max, wp_all, x, y


# Choose game tic-tac-toe with arbitrary players
game = tictactoe_with_arbitrary_players()

### NETWORK ARCHITECTURE ###
def network(state, nx, ny):
    # Set variable initializers
    init_weight = tf.random_normal_initializer(stddev = 0.1)
    init_bias = tf.constant_initializer(0.1)

    # Create variables "weights1" and "biases1".
    weights1 = tf.get_variable("weights1", [3, 3, 3, 30], initializer = init_weight)
    biases1 = tf.get_variable("biases1", [30], initializer = init_bias)

    # Create 1st layer
    conv1 = tf.nn.conv2d(state, weights1, strides = [1, 1, 1, 1], padding = 'SAME')
    out1 = tf.nn.relu(conv1 + biases1)

    # Create variables "weights2" and "biases2".
    weights2 = tf.get_variable("weights2", [3, 3, 30, 50], initializer = init_weight)
    biases2 = tf.get_variable("biases2", [50], initializer = init_bias)

    # Create 2nd layer
    conv2 = tf.nn.conv2d(out1, weights2, strides = [1, 1, 1, 1], padding ='SAME')
    out2 = tf.nn.relu(conv2 + biases2)

    # Create variables "weights1fc" and "biases1fc".
    weights1fc = tf.get_variable("weights1fc", [nx * ny * 50, 100], initializer = init_weight)
    biases1fc = tf.get_variable("biases1fc", [100], initializer = init_bias)

    # Create 1st fully connected layer
    fc1 = tf.reshape(out2, [-1, nx * ny * 50])
    out1fc = tf.nn.relu(tf.matmul(fc1, weights1fc) + biases1fc)

    # Create variables "weights2fc" and "biases2fc".
    weights2fc = tf.get_variable("weights2fc", [100, 3], initializer = init_weight)
    biases2fc = tf.get_variable("biases2fc", [3], initializer = init_bias)

    # Create 2nd fully connected layer
    return tf.matmul(out1fc, weights2fc) + biases2fc


# Input (common for all networks)
S = tf.placeholder(tf.float32, shape = [None, game.nx, game.ny, 3], name = "S")

# network for loading from .ckpt
scope = "network"
with tf.variable_scope(scope):
    # Estimation for unnormalized log probability
    Y = network(S, game.nx, game.ny)
    # Estimation for probability
    P = tf.nn.softmax(Y, name = "softmax")


saver = tf.train.Saver()

with tf.Session() as sess:
    ### DEFAULT SESSION ###
    sess.as_default()

    ### VARIABLE INITIALIZATION ###
    sess.run(tf.global_variables_initializer())
    
    if len(sys.argv) != 2:
        print('Usaeg) python task1_sol.py checkpointfilename')
    else:
        saver.restore(sess, sys.argv[1])
        n_test = 8 * 6 * 4 * 2
        s = game.play_games(P, np.zeros((n_test)), [], np.arange(n_test), n_test, nargout = 1)
        win=s[0][0]; loss=s[0][1]; tie=s[0][2]
        print('net plays black: win %f, loss %f, tie %f' % (win, loss, tie))
        n_test = 9 * 7 * 5 * 3
        s = game.play_games([], np.arange(n_test), P, np.zeros((n_test)), n_test, nargout = 1)
        win=s[0][1]; loss=s[0][0]; tie=s[0][2]
        print('net plays white: win %f, loss %f, tie %f' % (win, loss, tie))

