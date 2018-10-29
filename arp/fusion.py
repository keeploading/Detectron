# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import optimize
import math
import logging
import sys
import random
import numpy as np
import pylab as pl
from pykalman import KalmanFilter

STEER_RATIO = 13.0
WHEEL_BASE = 2.70

class Fusion(object):
    def __init__(self):
        random_state = np.random.RandomState(0)
        t = 0.1
        self.transition_matrix = np.array([[1, t, 0.5*(t**2)], [0, 1,t], [0,0,1]])
        self.transition_offset = np.array([0, 0, 0])
        self.observation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) #np.eye(3)# - random_state.randn(3, 3) * 0.01
        self.observation_offset = np.array([0, 0, 0])
        self.transition_covariance = np.array([[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.0001]])#np.eye(3)#
        self.observation_covariance = np.array([[0.5, 0, 0], [0, 0.3, 0], [0, 0, 0.001]])#np.eye(3)#
        self.initial_state_mean = np.array([0, 1, 0])
        self.state = [self.initial_state_mean, self.initial_state_mean]
        self.initial_state_covariance = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.kf = KalmanFilter(
            self.transition_matrix, self.observation_matrix, self.transition_covariance,
            self.observation_covariance, self.transition_offset, self.observation_offset,
            self.initial_state_mean, self.initial_state_covariance,
            random_state=random_state
        )
    def sample(self):
        pass

    def update(self, x_pos, v_speed, a_acc):
        X = [x_pos, v_speed, a_acc]
        print ("observation X:" + str(X))
        print ("self.transition_covariance:" + str(self.transition_covariance))
        self.transition_covariance[0, 0] = 0.0001#random.randint(0, 100) * 0.00002
        self.transition_covariance[1, 1] = random.randint(0, 100) * 0.001
        self.transition_covariance[2, 2] = random.randint(0, 100) * 0.00001

        self.observation_covariance[0, 0] = 0.5#random.randint(0, 100) * 0.01
        self.observation_covariance[1, 1] = random.randint(0, 100) * 0.003
        self.observation_covariance[2, 2] = random.randint(0, 100) * 0.00002
        self.state = self.kf.filter_update(self.state[0], self.state[1], X, self.transition_matrix,
                                     self.transition_offset, self.transition_covariance, self.observation_matrix,
                                     self.observation_offset, self.observation_covariance)
        print ("predict X:" + str(self.state[0]))
        return self.state

def test():
    fusion = Fusion()
    initial_state_mean = [0, 1, 0.05]
    # states, observations = fusion.kf.sample(
    #     n_timesteps=100,
    #     initial_state=initial_state_mean
    # )
    states, observations = [], []
    for i in range(100):
        i = i / 10.
        s = [0 + initial_state_mean[1]*i + 0.5 * initial_state_mean[2] *i**2, initial_state_mean[1] + initial_state_mean[2]*i, initial_state_mean[2]]
        states.append(s)
        observations.append([s[0] + random.randint(-30, 30) * 0.1, s[1] + random.randint(-30, 30) * 0.01, s[2] + random.randint(-30, 30) * 0.001])

    states = np.array(states)
    observations = np.array(observations)

    log_result = []
    log_obs = []
    for observation in observations:
        ret = fusion.update(observation[0], observation[1], observation[2])
        log_result.append(ret[0][0])
        log_obs.append(observation[0])

    log_result = np.array(log_result)
    log_obs = np.array(log_obs)
    # draw estimates
    pl.figure()
    lines_true = pl.plot(states[:,0], color='r')
    print("log_obs:" + str(log_obs.shape))
    lines_obs = pl.plot(log_obs, 'bo')
    lines_predict = pl.plot(log_result, color='g')
    pl.legend((lines_true[0], lines_obs[0], lines_predict[0]),
              ('true', 'obs', 'filt'),
              loc='lower right'
              )
    pl.show()

test()