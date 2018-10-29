# coding: utf-8
#  Copyright (c) 2017-present, Facebook, Inc.
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
import time
import numpy as np
import pylab as pl
from pykalman import UnscentedKalmanFilter
from numpy.linalg import cholesky
from arp.line_detection import lane_wid, IMAGE_HEI


#state (x y v theta w)
#(v/?)sin(??t+?)?(v/?)sin(?)+x(t)
#?(v/?)cos(??t+?)+(v/?)cos(?)+y(t)
#? to x axis

class Fusion(object):
    def __init__(self, x, v, w):
        self.random_state = np.random.RandomState(0)
        self.transition_covariance = np.array([[0.5, 0, 0, 0, 0],\
                                                [0, 1, 0, 0, 0],\
                                                [0, 0, 0.1, 0, 0],\
                                                [0, 0, 0, 0.001, 0],\
                                                [0, 0, 0, 0, 0.001],\
                                                ])
        self.observation_covariance = np.array([[0.5, 0, 0, 0, 0],\
                                               [0, 1, 0, 0, 0],\
                                               [0, 0, 0.5, 0, 0],\
                                               [0, 0, 0, 0.001, 0],\
                                               [0, 0, 0, 0, 0.001],\
                                              ])
        self.initial_state_mean = [x, 0.1, v, np.pi / 2, w]
        # self.initial_state_mean = [0, 0, 20, 0, np.pi / 180]
        self.transition_state = self.initial_state_mean
        self.obs = self.initial_state_mean
        self.pre_parabola_param = [0, 0, 1.75]
        self.initial_state_covariance = np.array([[0.5, 0, 0, 0, 0],\
                                                  [0, 0.02, 0, 0, 0],\
                                                  [0, 0, 0.1, 0, 0],\
                                                  [0, 0, 0, 0.001, 0],\
                                                  [0, 0, 0, 0, 0.001],\
                                                  ])
        self.T = 0.5
        self.estimate_state = [self.initial_state_mean, self.initial_state_covariance]
        self.kf = UnscentedKalmanFilter(
            self.transition_function, self.observation_function,
            self.transition_covariance, self.observation_covariance,
            self.initial_state_mean, self.initial_state_covariance,
            random_state=self.random_state
        )
        self.timestamp = time.time()

    def transition_function(self, state, noise):
        t = self.T
        if state[4] == 0:
            a = state[2] * np.cos(state[3]) + state[0]
            b = state[2] * np.sin(state[3]) + state[1]
            c = state[2]
            d = np.pi / 2#state[4] * t + state[3]
            e = state[4]
        else:
            r = state[2] / state[4]
            a = r * np.sin(state[4] * t + state[3]) - r * np.sin(state[3])# + state[0]
            b = -r * np.cos(state[4] * t + state[3]) + r * np.cos(state[3]) + state[1]
            c = state[2]
            d = np.pi / 2#state[4] * t + state[3]
            e = state[4]# + np.sin(debug_index / 10) * 0.01
            # e = states_i + np.sin(debug_index / 10) * 10 * np.pi / 180
        #adjust x from parabola param(down-right coordinate)
        pixl_b = b * (IMAGE_HEI / 40)
        parabola_y1 = self.pre_parabola_param[0] * (-pixl_b)**2 + self.pre_parabola_param[1] * (-pixl_b) + self.pre_parabola_param[2]
        dalta_y = parabola_y1 - self.pre_parabola_param[2]
        # a = dalta_y * (3.5 / lane_wid) + a
        pixl_y = parabola_y1 * (3.5 / lane_wid)
        a = pixl_y - a
        b = b - state[1]
        if self.obs[0] - a > 3.5/2:
            a += 3.5
        elif self.obs[0] - a < -3.5/2:
            a -= 3.5
        self.transition_state = np.array([a, b, c, d, e]) + noise
        # print ("self.transition_state:" + str(self.transition_state))
        return self.transition_state

    def observation_function(self, state, noise):
        # C = np.array([[-1, 0.5], [0.2, 0.1]])
        # C = np.array([[1, 0], [0, 1]])
        C = np.eye(5)
        # return np.dot(C, state) + noise
        return state + noise

    def update(self, obs):
        self.estimate_state = self.kf.filter_update(self.estimate_state[0], self.estimate_state[1], obs, self.transition_function,
                                     self.transition_covariance, self.observation_function, self.observation_covariance)
        print ("estimate X:" + str(self.estimate_state[0]))
        return self.estimate_state

    def update_step(self, x, v, w, t, parabola_param):
        print ("update_step1:({},{},{},{},{})".format(x,v,w,t,str(parabola_param)))
        self.T = t
        self.parabola_param = parabola_param
        # obs = [x, y, v, theta, w]
        # we didn't have obs_y so use predict obs_y(we didn't use y)
        if w == 0 or self.transition_state[4] == 0:
            y = self.transition_state[2] * np.sin(self.transition_state[3]) + self.transition_state[1]
        else:
            r = self.transition_state[2] / self.transition_state[4]
            y = -r * np.cos(self.transition_state[4] * t + self.transition_state[3]) + r * np.cos(self.transition_state[3]) + self.transition_state[1]
        self.obs = [x, y, v, np.pi / 2, w]
        self.timestamp = time.time()
        print ("update_step2:({})".format(str(self.obs)))
        self.update(self.obs)
        self.pre_parabola_param = parabola_param
        print ("update_step3:({})".format(str(self.estimate_state[0])))
        return self.estimate_state[0][0]
    def get_estimate(self):
        return self.estimate_state[0][0]
    def get_predict(self):
        return self.transition_state[0]




def test():
    fusion = Fusion()
    # states, observations = fusion.kf.sample(50, fusion.initial_state_mean)

    states, observations = [], []
    states_init = fusion.initial_state_mean
    filtered_state_estimates = []

    for i in range(1000):
        # i = i / 10.
        i = fusion.T
        if states_init[4] == 0:
            a = states_init[2] * np.cos(states_init[3]) + states_init[0]
            b = states_init[2] * np.sin(states_init[3]) + states_init[1]
            c = states_init[2]
            d = states_init[4] * i + states_init[3]
            e = states_init[4]
        else:
            # r = abs(states_init[2] / states_init[4])
            r = states_init[2] / states_init[4]
            a = r * np.sin(states_init[4] * i + states_init[3]) - r * np.sin(states_init[3]) + states_init[0]
            b = -r * np.cos(states_init[4] * i + states_init[3]) + r * np.cos(states_init[3]) + states_init[1]
            c = states_init[2]
            d = states_init[4] * i + states_init[3]
            e = states_init[4]# + np.sin(debug_index / 10) * 0.01
        if d > 2 * np.pi:
            d = d - 2 * np.pi
        elif d < 0:
            d = d + 2 * np.pi

        true_mu = np.array([[a, b]])
        true_sigma = np.array([[2, 0], [0, 4]])
        true_R = cholesky(true_sigma)
        true_p = np.dot(np.random.randn(1, 2), true_R) + true_mu
        true_v = 0.1 * np.random.randn(1) + c
        true_theta = 0.1 * np.random.randn(1) + d
        true_w = 0.1 * np.random.randn(1) + e
        states_init = [a,b,c,d,e]
        states_guss = [true_p[0][0], true_p[0][1], true_v[0], true_theta[0], true_w[0]]
        states.append(states_guss)
        # states.append(states_init)

        mu = np.array([[a, b]])
        Sigma = np.array([[10, 0], [0, 50]])
        R = cholesky(Sigma)
        p = np.dot(np.random.randn(1, 2), R) + mu
        v = 0.1 * np.random.randn(1) + c
        theta = 0.1 * np.random.randn(1) + d
        w = 0.1 * np.random.randn(1) + e

        obs = [p[0][0], p[0][1], v[0], theta[0], w[0]]
        observations.append(obs)
        filtered_state_estimates.append(fusion.update(obs)[0])
    states = np.array(states)
    observations = np.array(observations)
    filtered_state_estimates = np.array(filtered_state_estimates)
    # estimate state with filtering and smoothing
    # filtered_state_estimates = fusion.kf.filter(observations)[0]
    # smoothed_state_estimates = kf.smooth(observations)[0]

    # draw estimates
    pl.figure()
    lines_true = pl.plot(states[:, 0:2], color='b')
    lines_filt = pl.plot(filtered_state_estimates[:, 0:2], color='r', ls='-')
    point_obs = pl.plot(observations[:, 0:2], 'go')
    pl.legend((lines_true[0], lines_filt[0], point_obs[0]),
              ('true', 'filt', 'point_obs'),
              loc='lower left'
              )
    pl.show()

# test()

# debug_index = 0
# for i in range(50):
#     debug_index += (np.pi / 6)
#     print ("np.sin(debug_index / 10):" + str(np.sin(debug_index / 10)))
