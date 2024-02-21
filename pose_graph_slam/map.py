import numpy as np
import matplotlib.pyplot as plt

from icp import *
from util import *
from sensor import *
from geometry import *

        
class Map2D():

    def __init__(self):
        self.line_list = []
        self.N_line = 0

    def add_line(self, pt1, pt2):
        line = Line(pt1, pt2)

        self.line_list.append(line)
        self.N_line += 1
    
    def gen_map(self):
        #define outer contour
        O_pt = Pt(0, 0)
        S_pt, E_pt = O_pt, O_pt.add(10, 0)
        self.add_line(S_pt, E_pt)
        S_pt, E_pt = E_pt, E_pt.add(0, 5)
        self.add_line(S_pt, E_pt)
        S_pt, E_pt = E_pt, E_pt.add(1.5, 0)
        self.add_line(S_pt, E_pt)

        #define door at outer contour
        O_pt = Pt(13.5, 5)
        S_pt, E_pt = O_pt, O_pt.add(1.5, 0)

        #continue to define outer contour
        self.add_line(S_pt, E_pt)
        S_pt, E_pt = E_pt, E_pt.add(0, 5)
        self.add_line(S_pt, E_pt)
        S_pt, E_pt = E_pt, E_pt.add(-15, 0)
        self.add_line(S_pt, E_pt)
        S_pt, E_pt = E_pt, E_pt.add(0, -10)
        self.add_line(S_pt, E_pt)

        #define room 1
        O_pt = Pt(0, 5)
        S_pt, E_pt = O_pt, O_pt.add(5, 0)
        self.add_line(S_pt, E_pt)
        S_pt, E_pt = E_pt, E_pt.add(0, -1.5)
        self.add_line(S_pt, E_pt)
        O_pt = Pt(5, 0)
        S_pt, E_pt = O_pt, O_pt.add(0, 1.5)
        self.add_line(S_pt, E_pt)

        #define room 2
        O_pt = Pt(5, 5)
        S_pt, E_pt = O_pt, O_pt.add(0, 1.5)
        self.add_line(S_pt, E_pt)
        O_pt = Pt(5, 10)
        S_pt, E_pt = O_pt, O_pt.add(0, -1.5)
        self.add_line(S_pt, E_pt)

    def plot(self):
        ax = plt.subplot(1, 1, 1)

        for line in self.line_list:
            ax.plot([line.pt1.x, line.pt2.x],
                    [line.pt1.y, line.pt2.y], '-b')
        ax.set_aspect('equal', adjustable='box')
        plt.show()

class Occupancy_Map():

    def __init__(self, 
                 min_x = -5.,
                 min_y = -5.,
                 w = 25.,
                 l = 25.,
                 resol = 0.05):
        
        self.min_x = min_x
        self.min_y = min_y
        self.w = w
        self.l = l
        self.resol = resol if resol is not None else 0.1
        self.threshold = 5
        self.nr = int(l//self.resol + np.ceil(l%self.resol))
        self.nc = int(w//self.resol + np.ceil(w%self.resol))
        self.gmap     = np.zeros(self.nr*self.nc)

    def reset(self):
        self.gmap     = np.zeros(self.nr*self.nc)

    def w2g(self, x, y):

        dx = (x - self.min_x)
        dy = (y - self.min_y)
        r_ind = dy//self.resol + np.ceil(dy%self.resol)
        c_ind = dx//self.resol + np.ceil(dx%self.resol)

        ind = int(r_ind*self.nc + c_ind)
        return ind

    def g2w(self, g_ind):

        r_ind = g_ind//self.nr
        c_ind = g_ind % self.nr

        xg = c_ind*self.resol + self.resol/2. + self.min_x
        yg = r_ind*self.resol + self.resol/2. + self.min_y

        return xg, yg

    def update_grid(self, x, y):

        if x < self.min_x or y < self.min_y:
            return 
        
        if x > self.min_x + self.w or y > self.min_y + self.l:
            return

        ind = self.w2g(x, y)

        self.gmap[ind] += 1

    def extract_w_map(self):

        inds = (self.gmap >= self.threshold)
        w_map = np.array([])

        for i in range(len(self.gmap)):
            if inds[i]:
                gx, gy = self.g2w(i)

                w_map = np.vstack([w_map, np.array([gx, gy])]) if w_map.size else np.array([[gx, gy]])

        return w_map

        



