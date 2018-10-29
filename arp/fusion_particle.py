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
import json
import time
import numpy as np
import pylab as pl
from pykalman import UnscentedKalmanFilter
from numpy.linalg import cholesky
from arp.line_detection import lane_wid, IMAGE_HEI
import random
import bisect
import zmq
import cv2
import threading
import arp.const as const
from multiprocessing import Process, Queue


PARTICLE_COUNT = 500
LANE_WID = 3.5
RANDOM_RANGE = 1.75
np.random.seed(0)

def add_noise(level, *coords):
    return [x + random.uniform(-level, level) for x in coords]

def add_little_noise(*coords):
    return add_noise(0.02, *coords)

def add_some_noise(*coords):
    return add_noise(0.1, *coords)

class Particle(object):
    def __init__(self, x, y, heading=None, noisy=False):
        if noisy:
            x, heading = add_some_noise(x, heading)

        self.x = x
        self.h = heading
        self.w = 1
        self.x_move = 0
        self.y_move = 0
        self.move_time = 0
        self.y = y
        self.added = True

    def __repr__(self):
        return "(%f, %f)" % (self.x, self.w)
        # return "(%f, %f)" % (self.x, self.h)

    @property
    def xh(self):
        return self.x, self.h

    @classmethod
    def create_particle(cls, x, y, h, count):
        return [cls(Particle.create_normal(x), Particle.create_normal(y), heading=h) for _ in range(0, count)]

    @classmethod
    def create_particle_random(cls, x, y, h, count):
        return [cls(Particle.create_random(x), Particle.create_random(y), heading=h) for _ in range(0, count)]

    @classmethod
    def create_random(cls, x):
        return x + random.uniform(-RANDOM_RANGE, RANDOM_RANGE)

    @classmethod
    def create_normal(cls, x):
        return x + 0.5 * np.random.randn(1)[0]

    def read_sensor(self, x_r):
        return x_r - self.x

    def reset_by_origin(self, adjust_x, adjust_y, h):
        self.h = (self.h - h) + np.pi/2
        self.x += adjust_x
        self.y += adjust_y
        self.x_move = 0
        if self.y < -302 or self.y > 302:
            self.y = 0
            self.y_move = 0

    def move_by(self, x, y):
        self.x += x
        self.y += y

    def move_high_freq(self, speed, w, noisy=True):
        if self.move_time == 0:
            self.move_time = time.time()
            return
        if noisy:
            speed = add_noise(1, speed)[0]
            w = add_noise(0.02, w)[0]
        dalta_time = time.time() - self.move_time
        r = speed / w
        # print ("{} Particle move_high_freq:{}".format(time.time(), self.x_move))
        self.x_move += r * np.sin(self.h + dalta_time * w) - r * np.sin(self.h)
        self.y_move +=-r * np.cos(self.h + dalta_time * w) + r * np.cos(self.h)
        self.h += dalta_time * w
        self.move_time = time.time()
        # h += random.uniform(-3, 3)

class CarWayPoint(Particle):
    id = 0

    def __init__(self, index, x, y, h=0):
        super(CarWayPoint, self).__init__(Particle.create_normal(x), Particle.create_normal(y), heading=h)
        self.id = index
        self.step_count = 0
        self.h = h

    def read_sensor(self, x_right_line):
        """
        Poor robot, it's sensors are noisy and pretty strange,
        it only can measure the distance to the nearest beacon(!)
        and is not very accurate at that too!
        """
        # return add_noise(0.1, super(CarWayPoint, self).read_sensor(x_right_line))
        return super(CarWayPoint, self).read_sensor(x_right_line)

    def reset_by_origin(self, adjust_x):
        self.x = 0
        self.h = np.pi/2
        self.x_move = 0
        self.y = 0
        # if self.y < -302 or self.y > 302:
        #     self.y = 0
        #     self.y_move = 0
        print ("CarWayPoint reset{}".format(self.x_move))

    def info(self):
        return self.h, self.x_move, self.y_move


sigma2 = 1 ** 2
def w_gauss(a, b):
    error = a - b
    print ("w_gauss loop a:{}, b:{}:".format(a, b))
    g = math.e ** -(error ** 2 / (2 * sigma2))
    return g

def compute_mean_point(particles):
    """
    Compute the mean for all particles that have a reasonably good weight.
    This is not part of the particle filter algorithm but rather an
    addition to show the "best belief" for current position.
    """

    x, m_count = 0, 0
    for p in particles:
        m_count += p.w
        x += p.x * p.w
        # m_y += p.x_right * p.w

    if m_count == 0:
        print("m_count == 0")
        return 0, 1

    x /= m_count
    # m_y /= m_count

    # Now compute how good that mean is -- check how many particles
    # actually are in the immediate vicinity
    m_count = 0
    for p in particles:
        if abs(p.x - x) < 0.5:
            m_count += 1

    return x, (float(m_count) / len(particles))

class WeightedDistribution(object):
    def __init__(self, state):
        accum = 0.0
        self.state = [p for p in state if p.w > 2./PARTICLE_COUNT]
        # self.state = [p for p in state if p.w > 0.1]
        self.distribution = []
        for x in self.state:
            accum += x.w
            self.distribution.append(accum)

    def pick(self):
        try:
            return self.state[bisect.bisect_left(self.distribution, random.uniform(0, 1))]
        except IndexError:
            # Happens when all particles are improbable w=0
            return None

class FusionParticle(threading.Thread):
    def __init__(self, x_right_line, queue):
        threading.Thread.__init__(self)
        self.particles = Particle.create_particle_random(0, 0, np.pi/2, PARTICLE_COUNT)
        self.pre_x_right_line = x_right_line
        self.timestamp = time.time()
        self.queue = queue
        self.way_point = CarWayPoint(0, 0, 0, np.pi/2)

    def run(self):
        while(True):
            msg = self.queue.get(True)
            json_item = json.loads(msg)
            speed = json_item["speed"]
            angle = json_item["steerAngle"]
            if angle == 0:
                angle = random.uniform(-0.0001, 0.0001)
            wheel_theta = angle / const.STEER_RATIO
            wheel_theta = math.radians(wheel_theta)
            w = speed / (const.WHEEL_BASE / np.sin(wheel_theta))
            self.update_dr(speed, w)

    def update_dr(self, speed, w):
        self.way_point.move_high_freq(speed, w, noisy=False)
        for p in self.particles:
            p.move_high_freq(speed, w, noisy=False)

    def get_waypoint(self):
        return self.way_point

    def update(self, x_right_line):
        changed_to_left = False
        changed_to_right = False

        if x_right_line - self.pre_x_right_line > LANE_WID / 2:
            changed_to_right = True
            self.pre_x_right_line += LANE_WID
        elif x_right_line - self.pre_x_right_line < -LANE_WID/2:
            changed_to_left = True
            self.pre_x_right_line -= LANE_WID


        h, x_move, y_move = self.way_point.info()
        print ("FusionParticle update x_move:{}, h:{}".format(x_move, h - np.pi / 2))

        self.way_point.move_by(x_move, y_move)
        for p in self.particles:
            p.h = h  # in case robot changed heading, swirl particle heading too
            p.move_by(p.x_move, p.y_move)
            p.added = False

        # x_r = self.way_point.read_sensor(read_right_line)[0]
        x_r = self.pre_x_right_line - x_move

        # Update particle weight according to how good every particle matches
        # car's sensor reading
        for p in self.particles:
            # p_d = p.read_sensor(x_right_line)
            p_d = x_move + x_right_line - p.x
            p.w = w_gauss(x_r, p_d)
        # ---------- Try to find current best estimate for display ----------
        ret = compute_mean_point(self.particles)
        print ("self.particles:" + str(self.particles))
        print ("compute_mean_point ret:" + str(ret))
        x, m_confident = ret
        print ("current state2,x:{}".format(ret[0]))
        x_estimate = x_move + x_right_line - x
        x_change = (x - x_move)
        # ---------- Show current state ----------
        print ("current state, x_right_line:{}, self.pre_x_right_line:{}, x_estimate:{},x:{}".format(x_right_line, self.pre_x_right_line, x_estimate, x))

        # ---------- Shuffle particles ----------
        new_particles = []

        # Normalise weights
        nu = sum(p.w for p in self.particles)
        if nu:
            for p in self.particles:
                p.w = p.w / nu

        # create a weighted distribution, for fast picking
        dist = WeightedDistribution(self.particles)
        new_cnt = 0
        for _ in self.particles:
            p = dist.pick()
            if p is None:
                new_particle = Particle.create_particle(0, 0, h, 1)[0]
                new_cnt += 1
            else:
                new_particle = Particle(p.x, p.y, heading=self.way_point.h, noisy=True)
                new_particle.reset_by_origin(-p.x_move, -p.y_move, h)
                new_particle.added = False
            new_particles.append(new_particle)

        print ("new_particles x_move:{}, new_cnt:{}".format(str(x_move), new_cnt))
        self.particles = new_particles
        self.way_point.reset_by_origin(-x_move)
        self.timestamp = time.time()
        self.pre_x_right_line = x_right_line
        return x_estimate, self.particles


def dr_recever1():
    print ("dr recever process start !")
    sub_context = zmq.Context()
    socket = sub_context.socket(zmq.SUB)
    print ("tcp://localhost:{}".format(const.PORT_DR_OUT))
    socket.connect("tcp://localhost:{}".format(const.PORT_DR_OUT))
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    # socket.setsockopt(zmq.CONFLATE, 1)
    while(True):
        try:
            string = socket.recv()
            # print ("Received:{}".format(len(string)))
            if g_dr_queue.full():
                g_dr_queue.get(True)
            g_dr_queue.put(string)
        except zmq.ZMQError, Queue.em:
            time.sleep(1)


if __name__ == '__main__':
    x = 0
    g_dr_queue = Queue(10)
    g_particle_filter = FusionParticle(x, g_dr_queue)
    g_particle_filter.start()

    pdr_receiver = Process(target=dr_recever1)
    pdr_receiver.start()

    meter_scale = (3.5/100)
    while(True):
        time.sleep(0.1)
        img = np.zeros((604, 960, 3), np.uint8)+ 255
        points = np.array([[0, 302], [960,302], [480, 0], [480, 604]])
        cv2.polylines(img, np.int32([np.vstack((points[:,0], points[:,1])).T]), False, (0,0,0), thickness=1)
        x_measure = 1.75 + 3 * np.random.randn(1)[0]
        x_estimate, particles = g_particle_filter.update(x_measure)
        x_change = x_estimate - x_measure

        way_point = g_particle_filter.get_waypoint()
        for p in particles:
            color = abs(p.x - way_point.x) / meter_scale
            if p.added:
                color = (255,0,255)
            else:
                color = (50, 0, color % 255)
            cv2.circle(img, (int(p.x/meter_scale) * 5 + 480, int(p.y/meter_scale) + 302), 1, color, thickness=2)
        cv2.circle(img, (int(way_point.x/meter_scale) * 5 + 480, int(way_point.y) + 302), 4, (0, 0, 255))
        cv2.circle(img, (int((way_point.x+x_change)/meter_scale) + 480, int(way_point.y) + 302), 4, (0, 0, 255), thickness=8)
        print ("x_change meter:{}, pixls:{}".format(way_point.x+x_change, (way_point.x+x_change)/meter_scale))
        cv2.imshow('carlab', img)
        cv2.waitKey(1)

