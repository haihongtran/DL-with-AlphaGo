# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Project 2 - Task 2
# Tran Hong Hai

import numpy as np

# Spider Environment
class spider_environment:
    def __init__(self):
        self.n_states = 256
        self.n_actions = 256
        self.reward = np.zeros([self.n_states, self.n_actions])
        self.terminal = np.zeros(self.n_states, dtype=np.int)   # 1 if terminal state, 0 otherwise
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype=np.int)   # next_state
        transition = [[1,0,2,0],[1,0,3,1],[3,2,2,0],[3,2,3,1]]
        for s in range(self.n_states):
            for a in range(self.n_actions):
                total_down  = 0.
                total_force = 0.
                legStillDown = [0, 0, 0, 0]
                for i in range(4):  # 4 is number of legs
                    # Set next_state
                    s_eachLeg  = (s >> (2*i)) & 3
                    a_eachLeg  = (a >> (2*i)) & 3
                    sn_eachLeg = transition[s_eachLeg][a_eachLeg]
                    self.next_state[s,a] += (sn_eachLeg << (2*i))
                    # Set total_force
                    s_legUp = s_eachLeg & 1
                    s_legFw = (s_eachLeg >> 1) & 1
                    a_legFw = (a_eachLeg & 3) == 2
                    a_legBw = (a_eachLeg & 3) == 3
                    total_force += ((s_legUp == 0 and s_legFw == 1 and a_legBw == 1) - (s_legUp == 0 and s_legFw == 0 and a_legFw == 1))
                    # Set total_down
                    sn_legUp = sn_eachLeg & 1
                    legStillDown[i] = (s_legUp == 0 and sn_legUp == 0)
                    total_down += legStillDown[i]
                # Set reward
                if total_down == 0.:
                    self.reward[s,a] = 0
                elif total_down >= 3.:
                    self.reward[s,a] = total_force / total_down
                elif (total_down == 2) and ((legStillDown[0] == 1 and legStillDown[3] == 1)
                    or (legStillDown[1] == 1 and legStillDown[2] == 1)):
                    self.reward[s,a] = total_force / total_down
                else:
                    self.reward[s,a] = 0.25 * total_force / total_down
        self.init_state = 0b00001010    # initial state
