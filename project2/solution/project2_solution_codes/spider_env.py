# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/12/2017

import numpy as np

# spider environment
class spider_environment: 
    def __init__(self):
        self.n_states = 256       # number of states: front left (FL) leg up/down, front left (FL) leg forward/backward, ...
        self.n_actions = 256      # number of actions 
        self.reward = np.zeros([self.n_states, self.n_actions])
        self.terminal = np.zeros(self.n_states, dtype=np.int)          # 1 if terminal state, 0 otherwise
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype=np.int)        # next_state
        self.init_state = 0b00001010        # initial state
        transition = [[1,0,2,0],[1,0,3,1],[3,2,2,0],[3,2,3,1]]
        for s in range(256):
            fl_up = s & 1
            fl_fw = (s >> 1) & 1
            fr_up = (s >> 2) & 1
            fr_fw = (s >> 3) & 1
            bl_up = (s >> 4) & 1
            bl_fw = (s >> 5) & 1
            br_up = (s >> 6) & 1
            br_fw = (s >> 7) & 1
            for a in range(256):
                action_fl_up = (a & 3) == 0
                action_fl_dn = (a & 3) == 1
                action_fl_fw = (a & 3) == 2
                action_fl_bw = (a & 3) == 3
                action_fr_up = ((a >> 2) & 3) == 0
                action_fr_dn = ((a >> 2) & 3) == 1
                action_fr_fw = ((a >> 2) & 3) == 2
                action_fr_bw = ((a >> 2) & 3) == 3
                action_bl_up = ((a >> 4) & 3) == 0
                action_bl_dn = ((a >> 4) & 3) == 1
                action_bl_fw = ((a >> 4) & 3) == 2
                action_bl_bw = ((a >> 4) & 3) == 3
                action_br_up = ((a >> 6) & 3) == 0
                action_br_dn = ((a >> 6) & 3) == 1
                action_br_fw = ((a >> 6) & 3) == 2
                action_br_bw = ((a >> 6) & 3) == 3
                self.next_state[s,a] = transition[s & 3][a & 3] | transition[(s >> 2) & 3][(a >> 2) & 3] << 2 | \
                    transition[(s >> 4) & 3][(a >> 4) & 3] << 4 | transition[(s >> 6) & 3][(a >> 6) & 3] << 6
                total_force = (fl_up == 0 and fl_fw == 1 and action_fl_bw == 1) - (fl_up == 0 and fl_fw == 0 and action_fl_fw == 1) \
                            + (fr_up == 0 and fr_fw == 1 and action_fr_bw == 1) - (fr_up == 0 and fr_fw == 0 and action_fr_fw == 1) \
                            + (bl_up == 0 and bl_fw == 1 and action_bl_bw == 1) - (bl_up == 0 and bl_fw == 0 and action_bl_fw == 1) \
                            + (br_up == 0 and br_fw == 1 and action_br_bw == 1) - (br_up == 0 and br_fw == 0 and action_br_fw == 1)
                fl_down = fl_up == 0 and action_fl_up == 0   # check if touching floor during the current step
                fr_down = fr_up == 0 and action_fr_up == 0
                bl_down = bl_up == 0 and action_bl_up == 0
                br_down = br_up == 0 and action_br_up == 0
                total_down = fl_down + fr_down + bl_down + br_down
                if total_down > 0:
                    if fl_down and br_down and not fr_down and not bl_down or \
                       fr_down and bl_down and not fl_down and not br_down:
                        self.reward[s,a] = 1. * total_force / total_down
                    elif total_down >= 3:
                        self.reward[s,a] = 1. * total_force / total_down
                    else:
                        self.reward[s,a] = .25 * total_force / total_down

 
