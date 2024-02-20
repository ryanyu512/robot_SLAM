import numpy as np
from sensor import *
from util import *

class Robot():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.v = 0
        self.yaw = 0
        self.theta_dot = 0

        self.lidar = Lidar()
        self.pts_B = []
        self.pts_G = []

    def init_state(self, x, y, yaw, v, theta_dot):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.theta_dot = theta_dot

    def update(self, dt, acc):

        self.v += acc*dt
        self.yaw = wrap_ang(self.yaw + self.theta_dot*dt)
        self.x += self.v*np.cos(self.yaw)*dt
        self.y += self.v*np.sin(self.yaw)*dt

    def measure(self, map):
        
        self.lidar.emit(self.x, self.y, self.yaw)
        self.pts_B, self.pts_G = self.lidar.receive(map, self.x, self.y, self.yaw)

    def get_state(self):
        s = np.array([self.x, self.y, self.yaw, self.v])
        s.shape = (4, 1)
        return s