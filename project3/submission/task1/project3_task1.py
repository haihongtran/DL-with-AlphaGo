# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Project 3 - Task 1
# Name: Hong Hai Tran

import tensorflow as tf
import numpy as np
from scipy.signal import convolve2d as conv2d
import sys

# Define game
class board_game(object):
    def next_move(self, b, state, game_in_progress, net, rn, p, move, nlevels = 1, rw = 0):
        # returns next move by using neural networks
        # this is a parallel version, i.e., returns next moves for multiple games
        # Input arguments: b,state,game_in_progress,net,rn,p,move,nlevels,rw
        #   b: current board states for multiple games
        #   state: extra states
        #   game_in_progress: 1 if game is in progress, 0 if ended
        #   net: neural network. can be empty (in that case 'rn' should be 1)
        #   rn: if 0 <= rn <= 1, controls randomness in each move (0: no randomness, 1: pure random)
        #     if rn = -1, -2, ..., then the first |rn| moves are random
        #   p: current player (1: black, 2: white)
        #   move: k-th move (1,2,3,...)
        #   nlevels (optional): tree search depth (1,2, or 3). default=1
        #     if nlevels is even, then 'net' should be the opponent's neural network
        #   rw (optional): randomization in calculating winning probabilities, default=0
        # Return values
        # new_board,new_state,valid_moves,wp_max,wp_all,x,y=next_move(b,game_in_progress,net,rn,p,move)
        #   new_board: updated board states containing new moves
        #   new_state: updated extra states
        #   n_valid_moves: number of valid moves
        #   wp_max: best likelihood of winning
        #   wp_all: likelihood of winning for all possible next moves
        #   x: x coordinates of the next moves in 'new_board'
        #   y: y coordinates of the next moves in 'new_board'
        
        # board size
        nx = self.nx; ny = self.ny; nxy = nx * ny
        # randomness for each game & minimum r
        r = rn; rmin = np.amin(r)
        # number of games
        if b.ndim>=3:
            ng = b.shape[2]
        else:
            ng=1
        # number of valid moves in each game 
        n_valid_moves = np.zeros((ng))
        # check whether each of up to 'nxy' moves is valid for each game
        valid_moves = np.zeros((ng, nxy))
        # win probability for each next move
        wp_all = np.zeros((nx, ny, ng))
        # maximum of wp_all over all possible next moves
        wp_max = -np.ones((ng))
        mx = np.zeros((ng))
        my = np.zeros((ng))
        x = -np.ones((ng))
        y = -np.ones((ng))

        # check nlevels
        if nlevels > 3 or nlevels <= 0:
            raise Exception('# of levels not supported. Should be 1, 2, or 3.')
        # total cases to consider in tree search
        ncases = pow(nxy, nlevels)

        # maximum possible board states considering 'ncases'
        d = np.zeros((nx, ny, 3, ng * ncases), dtype = np.int32)

        for p1 in range(nxy):
            vm1, b1, state1 = self.valid(b, state, self.xy(p1), p)
            n_valid_moves += vm1
            if rmin < 1:
                valid_moves[:, p1] = vm1
                if nlevels == 1:
                    c = 3 - p  # current player is changed to the next player after placing a stone at 'p1'
                    idx = np.arange(ng) + p1 * ng
                    d[:, :, 0, idx] = (b1 == c)     # 1 if current player's stone is present, 0 otherwise
                    d[:, :, 1, idx] = (b1 == 3 - c) # 1 if opponent's stone is present, 0 otherwise
                    d[:, :, 2, idx] = 2 - c         # 1: current player is black, 0: white
                else:
                    for p2 in range(nxy):
                        vm2, b2, state2 = self.valid(b1, state1, self.xy(p2), 3 - p)
                        if nlevels == 2:
                            c = p                 # current player is changed again after placing a stone at 'p2'
                            idx = np.arange((ng)) + p1 * ng + p2 * ng * nxy
                            d[:, :, 0, idx] = (b2 == c)
                            d[:, :, 1, idx] = (b2 == 3 - c)
                            d[:, :, 2, idx] = 2 - c
                        else:
                            for p3 in range(nxy):
                                vm3, b3, state3 = self.valid(b2, state2, self.xy(p3), p)
                                c = 3 - p         # current player is changed yet again after placing a stone at 'p3'
                                idx = np.arange(ng) + p1 * ng + p2 * ng * nxy\
                                        + p3 * ng * nxy * nxy
                                d[:, :, 0, idx] = (b3 == c)
                                d[:, :, 1, idx] = (b3 == 3 - c)
                                d[:, :, 2, idx] = 2 - c

        # n_valid_moves is 0 if game is not in progress
        n_valid_moves = n_valid_moves * game_in_progress

        # For operations in TensorFlow, load session and graph
        sess = tf.get_default_session()

        # d(nx, ny, 3, ng * ncases) becomes d(ng * ncases, nx, ny, 3)
        d = np.rollaxis(d, 3)
        if rmin < 1: # if not fully random, then use the neural network 'net'
            softout = np.zeros((d.shape[0], 3))
            size_minibatch = 1024
            num_batch = np.ceil(d.shape[0] / float(size_minibatch))
            for batch_index in range(int(num_batch)):
                batch_start = batch_index * size_minibatch
                batch_end = \
                        min((batch_index + 1) * size_minibatch, d.shape[0])
                indices = range(batch_start, batch_end)
                feed_dict = {'S:0': d[indices, :, :, :]}  # d[indices,:,:,:] goes to 'S' (neural network input)
                softout[indices, :] = sess.run(net, feed_dict = feed_dict) # get softmax output from 'net'
            if p == 1:   # if the current player is black
                # softout[:,0] is the softmax output for 'tie'
                # softout[:,1] is the softmax output for 'black win'
                # softout[:,2] is the softmax output for 'white win'
                wp = 0.5 * (1 + softout[:, 1] - softout[:, 2])  # estimated win prob. for black
            else:        # if the current player is white
                wp = 0.5 * (1 + softout[:, 2] - softout[:, 1])  # estimated win prob. for white

            if rw != 0:     # this is only for nlevels == 1
                # add randomness so that greedy action selection to be done later is randomized
                wp = wp + np.random.rand((ng, 1)) * rw

            if nlevels >= 3:
                wp = np.reshape(wp, (ng, nxy, nxy, nxy))
                wp = np.amax(wp, axis = 3)    

            if nlevels >= 2:
                wp = np.reshape(wp, (ng, nxy, nxy))
                wp = np.amin(wp, axis = 2)

            wp = np.transpose(np.reshape(wp,(nxy,ng)))
            wp = valid_moves * wp - (1 - valid_moves)
            wp_i = np.argmax(wp, axis = 1)  # greedy action selection
            mxy = self.xy(wp_i)             # convert to (x,y) coordinates

            for p1 in range(nxy):
                pxy = self.xy(p1)
                wp_all[int(pxy[:, 0]), int(pxy[:, 1]), :] = wp[:, p1]  # win prob. for each of possible next moves

        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b[:, :, :]
        new_state = np.zeros(state.shape)
        new_state[:, :] = state[:, :]

        # Choose action
        if rmin == 0:   # Net's turn
            for k in range(ng):
                if n_valid_moves[k]:    # If there are valid moves
                    isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], mxy[[k], :], p)
                    new_board[:, :, [k]] = bn
                    new_state[:, [k]] = sn
                    x[k] = mxy[k, 0]
                    y[k] = mxy[k, 1]
                else:   # No valid moves
                    isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], -np.ones((1, 2)), p)
                    new_state[:, [k]] = sn
        elif rmin == 1: # Opponent's turn
            # Opponent plays black
            if move == 1:   # First move
                i = j = 0; k = 0;
                cxy = np.zeros((1,2))
                for a0 in range(9):     # Iterating over 9 possible valid moves
                    if n_valid_moves[k]:    # If there are valid moves
                        while(1):   # find a possible move
                            cxy[0,:] = [i,j]
                            isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], cxy, p)
                            if isvalid:
                                if j != 2:
                                    j += 1
                                else:
                                    i += 1; j = 0;
                                break
                            else:
                                if j != 2:
                                    j += 1
                                else:
                                    i += 1; j = 0;
                        for a1 in range(7*5*3): # apply move cxy to 7*5*3 games
                            isvalid, bn, sn = self.valid(b[:, :, [k + a1]], state[:, [k + a1]], cxy, p)
                            new_board[:, :, [k + a1]] = bn
                            new_state[:, [k + a1]] = sn
                            x[k + a1] = cxy[0,0]
                            y[k + a1] = cxy[0,1]
                    else:   # No valid moves
                        for a1 in range(7*5*3):
                            isvalid, bn, sn = self.valid(b[:, :, [k + a1]], state[:, [k + a1]], -np.ones((1, 2)), p)
                            new_state[:, [k + a1]] = sn
                    k += 7*5*3
            elif move == 3:     # 3rd move
                k = 0
                cxy = np.zeros((1,2))
                for a0 in range(9):
                    i = j = 0;
                    for a1 in range(7): # Iterating over 7 possible valid moves
                        if n_valid_moves[k]:
                            while(1):   # Find a possible move
                                cxy[0,:] = [i,j]
                                isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], cxy, p)
                                if isvalid:
                                    if j != 2:
                                        j += 1
                                    else:
                                        i += 1; j = 0;
                                    break
                                else:
                                    if j != 2:
                                        j += 1
                                    else:
                                        i += 1; j = 0;
                            for a2 in range(5*3):   # Apply move cxy to 5*3 games
                                isvalid, bn, sn = self.valid(b[:, :, [k + a2]], state[:, [k + a2]], cxy, p)
                                new_board[:, :, [k + a2]] = bn
                                new_state[:, [k + a2]] = sn
                                x[k + a2] = cxy[0,0]
                                y[k + a2] = cxy[0,1]
                        else:
                            for a2 in range(5*3):
                                isvalid, bn, sn = self.valid(b[:, :, [k + a2]], state[:, [k + a2]], -np.ones((1, 2)), p)
                                new_state[:, [k + a2]] = sn
                        k += 5*3
            elif move == 5: # 5th move
                k = 0
                cxy = np.zeros((1,2))
                for a0 in range(9):
                    for a1 in range(7):
                        i = j = 0
                        for a2 in range(5): # Iterating over 5 possible moves
                            if n_valid_moves[k]:
                                while(1):   # find a possible move
                                    cxy[0,:] = [i,j]
                                    isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], cxy, p)
                                    if isvalid:
                                        if j != 2:
                                            j += 1
                                        else:
                                            i += 1; j = 0;
                                        break
                                    else:
                                        if j != 2:
                                            j += 1
                                        else:
                                            i += 1; j = 0;
                                for a3 in range(3): # Apply move cxy to 3 games
                                    isvalid, bn, sn = self.valid(b[:, :, [k + a3]], state[:, [k + a3]], cxy, p)
                                    new_board[:, :, [k + a3]] = bn
                                    new_state[:, [k + a3]] = sn
                                    x[k + a3] = cxy[0,0]
                                    y[k + a3] = cxy[0,1]
                            else:
                                for a3 in range(3):
                                    isvalid, bn, sn = self.valid(b[:, :, [k + a3]], state[:, [k + a3]], -np.ones((1, 2)), p)
                                    new_state[:, [k + a3]] = sn
                            k += 3
            elif move == 7: # 7th move
                k = 0
                cxy = np.zeros((1,2))
                for a0 in range(9):
                    for a1 in range(7):
                        for a2 in range(5):
                            i = j = 0
                            for a3 in range(3): # Iterate over 3 possible moves
                                if n_valid_moves[k]:
                                    while(1):   # Find possible move
                                        cxy[0,:] = [i,j]
                                        isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], cxy, p)
                                        if isvalid:
                                            if j != 2:
                                                j += 1
                                            else:
                                                i += 1; j = 0;
                                            break
                                        else:
                                            if j != 2:
                                                j += 1
                                            else:
                                                i += 1; j = 0;
                                    for a4 in range(1): # Apply move cxy to 1 game
                                        isvalid, bn, sn = self.valid(b[:, :, [k + a4]], state[:, [k + a4]], cxy, p)
                                        new_board[:, :, [k + a4]] = bn
                                        new_state[:, [k + a4]] = sn
                                        x[k + a4] = cxy[0,0]
                                        y[k + a4] = cxy[0,1]
                                else:
                                    for a4 in range(1):
                                        isvalid, bn, sn = self.valid(b[:, :, [k + a4]], state[:, [k + a4]], -np.ones((1, 2)), p)
                                        new_state[:, [k + a4]] = sn
                                k += 1
            elif move == 9: # Last move (last empty cell in the 9 cells of the board game)
                for k in range(ng):
                    cxy = np.zeros((1,2))
                    if n_valid_moves[k]:
                        i = j = 0
                        while(1):   # Finding possible move
                            cxy[0,:] = [i,j]
                            isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], cxy, p)
                            if isvalid:
                                break
                            else:
                                if j != 2:
                                    j += 1
                                else:
                                    i += 1; j = 0;
                                continue
                        isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], cxy, p)
                        new_board[:, :, [k]] = bn
                        new_state[:, [k]] = sn
                        x[k] = cxy[0,0]
                        y[k] = cxy[0,1]
                    else:
                        isvalid, bn, sn = self.valid(b[:, :, [k]], state[:, [k]], -np.ones((1, 2)), p)
                        new_state[:, [k]] = sn

            # Opponent plays white
            elif move == 2:     # 2nd move
                k = 0
                cxy = np.zeros((1,2))
                i = j = 0
                for a0 in range (8):    # Iterate over 8 possible moves
                    if n_valid_moves[k]:
                        while(1):   # Finding a possible move
                            cxy[0,:] = [i,j]
                            isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], cxy, p)
                            if isvalid:
                                if j != 2:
                                    j += 1
                                else:
                                    i += 1; j = 0;
                                break
                            else:
                                if j != 2:
                                    j += 1
                                else:
                                    i += 1; j = 0;
                        for a1 in range(6*4*2): # apply move cxy to 6*4*2 games
                            isvalid, bn, sn = self.valid(b[:, :, [k + a1]], state[:, [k + a1]], cxy, p)
                            new_board[:, :, [k + a1]] = bn
                            new_state[:, [k + a1]] = sn
                            x[k + a1] = cxy[0,0]
                            y[k + a1] = cxy[0,1]
                    else:
                        for a1 in range(6*4*2):
                            isvalid, bn, sn = self.valid(b[:, :, [k + a1]], state[:, [k + a1]], -np.ones((1, 2)), p)
                            new_state[:, [k + a1]] = sn
                    k += 6*4*2
            elif move == 4:     # 4th move
                k = 0
                cxy = np.zeros((1,2))
                for a0 in range(8):
                    i = j = 0;
                    for a1 in range(6): # Iterate over 6 possible moves
                        if n_valid_moves[k]:
                            while(1):   # Find a possible move
                                cxy[0,:] = [i,j]
                                isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], cxy, p)
                                if isvalid:
                                    if j != 2:
                                        j += 1
                                    else:
                                        i += 1; j = 0;
                                    break
                                else:
                                    if j != 2:
                                        j += 1
                                    else:
                                        i += 1; j = 0;
                            for a2 in range(4*2):   # apply move cxy to 4*2 games
                                isvalid, bn, sn = self.valid(b[:, :, [k + a2]], state[:, [k + a2]], cxy, p)
                                new_board[:, :, [k + a2]] = bn
                                new_state[:, [k + a2]] = sn
                                x[k + a2] = cxy[0,0]
                                y[k + a2] = cxy[0,1]
                        else:
                            for a2 in range(4*2):
                                isvalid, bn, sn = self.valid(b[:, :, [k + a2]], state[:, [k + a2]], -np.ones((1, 2)), p)
                                new_state[:, [k + a2]] = sn
                        k += 4*2
            elif move == 6:     # 6th move
                k = 0
                cxy = np.zeros((1,2))
                for a0 in range(8):
                    for a1 in range(6):
                        i = j = 0
                        for a2 in range(4):     # Iterate over 4 possible moves
                            if n_valid_moves[k]:
                                while(1):   # Find a possible move
                                    cxy[0,:] = [i,j]
                                    isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], cxy, p)
                                    if isvalid:
                                        if j != 2:
                                            j += 1
                                        else:
                                            i += 1; j = 0;
                                        break
                                    else:
                                        if j != 2:
                                            j += 1
                                        else:
                                            i += 1; j = 0;
                                for a3 in range(2):     # Apply move cxy to 2 games
                                    isvalid, bn, sn = self.valid(b[:, :, [k + a3]], state[:, [k + a3]], cxy, p)
                                    new_board[:, :, [k + a3]] = bn
                                    new_state[:, [k + a3]] = sn
                                    x[k + a3] = cxy[0,0]
                                    y[k + a3] = cxy[0,1]
                            else:
                                for a3 in range(2):
                                    isvalid, bn, sn = self.valid(b[:, :, [k + a3]], state[:, [k + a3]], -np.ones((1, 2)), p)
                                    new_state[:, [k + a3]] = sn
                            k += 2
            elif move == 8:     # Last move for white player
                k = 0
                cxy = np.zeros((1,2))
                for a0 in range(8):
                    for a1 in range(6):
                        for a2 in range(4):
                            i = j = 0
                            for a3 in range(2):     # Iterate over 2 possible moves
                                if n_valid_moves[k]:
                                    while(1):   # Find possible move
                                        cxy[0,:] = [i,j]
                                        isvalid, _, _ = self.valid(b[:, :, [k]], state[:, [k]], cxy, p)
                                        if isvalid:
                                            if j != 2:
                                                j += 1
                                            else:
                                                i += 1; j = 0;
                                            break
                                        else:
                                            if j != 2:
                                                j += 1
                                            else:
                                                i += 1; j = 0;
                                    for a4 in range(1):     # apply move cxy to 1 game
                                        isvalid, bn, sn = self.valid(b[:, :, [k + a4]], state[:, [k + a4]], cxy, p)
                                        new_board[:, :, [k + a4]] = bn
                                        new_state[:, [k + a4]] = sn
                                        x[k + a4] = cxy[0,0]
                                        y[k + a4] = cxy[0,1]
                                else:
                                    for a4 in range(1):
                                        isvalid, bn, sn = self.valid(b[:, :, [k + a4]], state[:, [k + a4]], -np.ones((1, 2)), p)
                                        new_state[:, [k + a4]] = sn
                                k += 1
            elif move == 10:
                pass
            else:
                print 'Unexpected move value. move =', move
        else:
            print 'Unexpected rmin value. rmin =', rmin

        return new_board, new_state, n_valid_moves, wp_max, wp_all, x, y


    def play_games(self, net1, r1, net2, r2, ng, max_time = 0, nargout = 1):
        # plays 'ng' games between two players
        # Inputs
        #   net1: neural network playing black. can be empty (r1 should be 1 if net1 is empty)
        #   r1: if 0 <= r1 <= 1, controls randomness in the next move by player 1 (0: no randomness, 1: pure random)
        #     if r1 = -1, -2, ..., then the first |r1| moves are random
        #   net2: neural network playing white. can be empty (r2 should be 1 if net2 is empty)
        #   r2: if 0 <= r2 <= 1, controls randomness in the next move by player 2 (0: no randomness, 1: pure random)
        #     if r2 = -1, -2, ..., then the first |r2| moves are random
        #   ng: number of games to play
        #   max_time (optional): the max. number of moves per game
        #   nargout (optional): the number of output arguments
        # Return values
        #   stat=play_games(net1,r1,net2,r2,ng,nargout=1): statistics for net1, stat=[win loss tie]
        #   d,w,wp,stat=play_games(net1,r1,net2,r2,ng,nargout=2,3, or 4)
        #     d: 4-d tensor of size nx*ny*3*nb containing all moves, where nb is the total number of board states
        #     w: nb*1, 0: tie, 1: black wins, 2: white wins
        #     wp (if nargout>=3):  win probabilities for the current player
        #     stat (if nargout==4): statistics for net1, stat=[win loss tie], for net2, swap win & loss
        
        
        # board size 
        nx = self.nx; ny = self.ny

        # maximum number of moves in each game
        if max_time <= 0:
            np0 = nx * ny * 2
        else:
            np0 = max_time

        # m: max. possible number of board states
        m = np0 * ng
        d = np.zeros((nx, ny, 3, m))
        
        # game outcome, tie(0)/black win(1)/white win(2), for each board state
        w = np.zeros((m))

        # winning probability       
        wp = np.zeros((m))

        # 1 means valid as training data, 0 means invalid
        valid_data = np.zeros((m))

        # current turn: 1 if black, 2 if white
        turn = np.zeros((m))

        # number of valid moves in the previous move
        vm0 = np.ones((ng))

        # initialize game
        if hasattr(self, 'game_init'): 
            [b, state] = self.game_init(ng)
        else:   # default initialization
            b = np.zeros((nx, ny, ng))
            state = np.zeros((0, ng))

        # maximum winning probability for each game
        wp_max = np.zeros((ng))

        # 1 if game is in progress, 0 otherwise
        game_in_progress = np.ones((ng))

        # first player is black (1)
        p = 1

        for k in range(np0):
            if p == 1:   # if black's turn, use net1 and r1
                b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                    self.next_move(b, state, game_in_progress, net1, r1, p, k + 1)
            else:        # if white's turn, use net2 and r2
                b, state, n_valid_moves, wp_max, _, x_pos, y_pos =\
                    self.next_move(b, state, game_in_progress, net2, r2, p, k + 1)

            # check if game ended and if winner is decided
            w0, end_game, _, _ = self.winner(b, state)
            idx = np.arange(k * ng, (k + 1) * ng)
            c = 3 - p    # current player is now changed to the next player
            d[:, :, 0, idx] = (b == c)      # 1 if current player's stone is present, 0 otherwise
            d[:, :, 1, idx] = (b == 3 - c)  # 1 if opponent's stone is present, 0 otherwise
            d[:, :, 2, idx] = 2 - c         # color of current player's stone (1 if black, 0 if white)
            
            wp[idx] = wp_max
            # valid as training data if game is in progress and if # of valid moves is > 0
            valid_data[idx] = game_in_progress * (n_valid_moves > 0)
            
            # information on the current player
            turn[idx] = p
            
            # update game_in_progress
            game_in_progress *=\
                    ((n_valid_moves > 0) * (end_game == 0) +\
                    ((vm0 + n_valid_moves) > 0) * (end_game == -1))
            # if end_game==1, game ends
            # if end_game==0, game ends if no more move is possible for the current player
            # if end_game==-1, game ends if no moves are possible for both players

            number_of_games_in_progress = np.sum(game_in_progress)
            if number_of_games_in_progress == 0:
                break   # no games to play

            p = 3 - p   # change the turn
            vm0 = n_valid_moves[:]  # preserve 'n_valid_moves'
                                    # no copying, which is ok since 'n_valid_moves' will be created as
                                    # a new array in the next step

        for k in range(np0):
            idx = np.arange(k * ng, (k + 1) * ng)
            w[idx] = w0[:] # final winner

        # player 1's stat
        win = np.sum(w0 == 1) / float(ng)
        loss = np.sum(w0 == 2) / float(ng)
        tie = np.sum(w0 == 0) / float(ng)

        varargout = []

        if nargout >= 2:
            fv = np.where(valid_data)[0]
            varargout.append(d[:, :, :, fv])
            varargout.append(w[fv])
            if nargout >= 3:
                varargout.append(wp[fv])
            if nargout >= 4:
                varargout.append([win, loss, tie])
        else:
            varargout.append([win, loss, tie])
        return varargout

    def xy(self, k): # xy position
        if hasattr(k, '__len__'):
            n = len(k)
        else:
            n = 1
        ixy = np.zeros((n, 2))
        ixy[:, 0] = np.floor(k / float(self.ny))
        ixy[:, 1] = np.mod(k, self.ny)
        return ixy

### TIC-TAC-TOE game ###
class tictactoe(board_game):
    def __init__(self, nx = 3, ny = 3, n = 3, name = 'tic tac toe', theme = 'basic'):
        self.nx = nx
        self.ny = ny
        self.n = n # n-mok
        self.name = name
        self.theme = theme

    def winner(self, b, state):
        # Check who wins for n-mok game
        # Inputs
        #    b: current board state, 0: no stone, 1: black, 2: white
        #    state: extra state
        # Usage) [r, end_game, s1, s2]=winner(b, state)
        #    r: 0 tie, 1: black wins, 2: white wins
        #    end_game
        #        if end_game==1, game ends
        #        if end_game==0, game ends if no more move is possible for the current player
        #        if end_game==-1, game ends if no moves are possible for both players
        #    s1: score for black
        #    s2: score for white

        # total number of games
        ng = b.shape[2]
        n = self.n
        r = np.zeros((ng))
        fh = np.ones((n, 1))
        fv = np.transpose(fh)
        fl = np.identity(n)
        fr = np.fliplr(fl)
        for j in range(ng):
            c = (b[:, :, j] == 1)
            if np.amax(conv2d(c, fh, mode = 'valid') == n)\
                or np.amax(conv2d(c, fv, mode = 'valid') == n)\
                or np.amax(conv2d(c, fl, mode = 'valid') == n)\
                or np.amax(conv2d(c, fr, mode = 'valid') == n):
                r[j] = 1

            c = (b[:, :, j] == 2)
            if np.amax(conv2d(c, fh, mode = 'valid') == n)\
                or np.amax(conv2d(c, fv, mode = 'valid') == n)\
                or np.amax(conv2d(c, fl, mode = 'valid') == n)\
                or np.amax(conv2d(c, fr, mode = 'valid') == n):
                r[j] = 2
        return r, r > 0, r == 1, r == 2

    def valid(self, b, state, xy, p):
        # Check if the move (x,y) is valid for a basic game where any empty board position is possible.
        # Inputs
        #    b: current board state, 0: no stone, 1: black, 2: white
        #    state: extra state
        #    xy=[xs, ys]: new position
        #    p: current player, 1 or 2
        # Return values
        #    [r,new_board,new_state]=valid(b,state,(xs,ys),p)
        #    r: 1 means valid, 0 means invalid
        #    new_board: updated board state
        #    new_state: updated extra state
        ng = b.shape[2]
        n = self.n
        if len(xy) < ng:
            xs = np.ones((ng)) * xy[:, 0]
            ys = np.ones((ng)) * xy[:, 1]
        else:
            xs = xy[:, 0]
            ys = xy[:, 1]

        # whether position is valid or not
        r = np.zeros((ng))
        new_board = np.zeros(b.shape)
        new_board[:, :, :] = b[:, :, :] # copy by values
        for j in range(ng):
            x = int(xs[j])
            y = int(ys[j])

            if x == -1 or y == -1:
                continue
            if b[x, y, j] == 0: # position is empty in the j-th game
                r[j] = 1
                new_board[x, y, j] = p 

        return r, new_board, state


# Declare tictactoe game
game = tictactoe()

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

# Network for loading from .ckpt
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
    
    # Load trained parameters
    saver.restore(sess, "project3_task1.ckpt")

    # Net plays black
    ng_test = 8 * 6 * 4 * 2
    r1 = np.zeros((ng_test)) # Player 1 uses greedy actions
    r2 = np.ones((ng_test))  # Player 2 uses random actions
    s = game.play_games(P, r1, [], r2, ng_test, nargout = 1)
    win1 = s[0][0]; lose1 = s[0][1]; tie1 = s[0][2];
    print(" Net plays black: win=%6.4f, loss=%6.4f, tie=%6.4f" %\
        (win1, lose1, tie1))

    # Net plays white
    ng_test = 9 * 7 * 5 * 3
    r1 = np.ones((ng_test))  # Player 1 uses random actions
    r2 = np.zeros((ng_test)) # Player 2 uses greddy actions
    s = game.play_games([], r1, P, r2, ng_test, nargout = 1)
    win2 = s[0][1]; lose2 = s[0][0]; tie2 = s[0][2];
    print(" Net plays white: win=%6.4f, loss=%6.4f, tie=%6.4f" %\
        (win2, lose2, tie2))
