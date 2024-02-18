import numpy as np
import matplotlib.pyplot as plt

class Pt():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, dx, dy):
        return Pt(self.x + dx, self.y + dy)

class Line():
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        self.compute_line_coeff()

    def compute_line_coeff(self):

        dx = float(self.pt2.x - self.pt1.x)
        dy = float(self.pt2.y - self.pt1.y)
        self.u = np.array([dx, dy])
        self.L = np.linalg.norm(self.u)

        self.u /= self.L
        self.P0 = self.pt1

class Lidar():

    def __init__(self, c):
        self.c = c
        self.max_range = 5.
        self.max_ang_range = np.math.pi*2.
        self.resol = np.deg2rad(2.)
        self.N_beams = int(self.max_ang_range/self.resol)
        self.beam_list = [None]*self.N_beams
        print(self.N_beams)

    def emit(self):
        
        emit_ang = np.linspace(0., 2*np.pi - self.resol, self.N_beams)
        print(emit_ang[0], np.rad2deg(emit_ang[-1]), np.rad2deg(emit_ang[1] - emit_ang[0]))

        xs = np.cos(emit_ang)*(-self.max_range) - np.sin(emit_ang)*0. + self.c.x
        ys = np.sin(emit_ang)*(-self.max_range) + np.cos(emit_ang)*0. + self.c.y

        cnt = 0
        for x, y in zip(xs, ys):
            self.beam_list[cnt] = Line(self.c, Pt(x, y))
            cnt += 1

class Map2D():

    def __init__(self):
        self.line_list = []
        self.N_line = 0

    def add_line(self, pt1, pt2):
        line = Line(pt1, pt2)

        self.line_list.append(line)
        self.N_line += 1
    
    def plot(self):
        ax = plt.subplot(1, 1, 1)

        for line in self.line_list:
            ax.plot([line.pt1.x, line.pt2.x],
                    [line.pt1.y, line.pt2.y], '-b')
        ax.set_aspect('equal', adjustable='box')
        plt.show()


def check_intersect(line1, line2):
    A = np.array([[line1.u[0], -line2.u[0]], 
                  [line1.u[1], -line2.u[1]]])
    b = np.array([[line2.P0.x - line1.P0.x], 
                  [line2.P0.y - line1.P0.y]])
    
    if np.linalg.det(A) != 0.:
        return np.matmul(np.linalg.inv(A), b)
    else:
        return 

map = Map2D()

#define outer contour
O_pt = Pt(0, 0)
S_pt, E_pt = O_pt, O_pt.add(10, 0)
map.add_line(S_pt, E_pt)
S_pt, E_pt = E_pt, E_pt.add(0, 5)
map.add_line(S_pt, E_pt)
S_pt, E_pt = E_pt, E_pt.add(1.5, 0)
map.add_line(S_pt, E_pt)

#define door at outer contour
O_pt = Pt(13.5, 5)
S_pt, E_pt = O_pt, O_pt.add(1.5, 0)

#continue to define outer contour
map.add_line(S_pt, E_pt)
S_pt, E_pt = E_pt, E_pt.add(0, 5)
map.add_line(S_pt, E_pt)
S_pt, E_pt = E_pt, E_pt.add(-15, 0)
map.add_line(S_pt, E_pt)
S_pt, E_pt = E_pt, E_pt.add(0, -10)
map.add_line(S_pt, E_pt)

#define room 1
O_pt = Pt(0, 5)
S_pt, E_pt = O_pt, O_pt.add(5, 0)
map.add_line(S_pt, E_pt)
S_pt, E_pt = E_pt, E_pt.add(0, -1.5)
map.add_line(S_pt, E_pt)
O_pt = Pt(5, 0)
S_pt, E_pt = O_pt, O_pt.add(0, 1.5)
map.add_line(S_pt, E_pt)

#define room 2
O_pt = Pt(5, 5)
S_pt, E_pt = O_pt, O_pt.add(0, 1.5)
map.add_line(S_pt, E_pt)
O_pt = Pt(5, 10)
S_pt, E_pt = O_pt, O_pt.add(0, -1.5)
map.add_line(S_pt, E_pt)

# map.plot()
# l1 = Line(Pt(0, 0), Pt(0, 10))
# l2 = Line(Pt(5, 5), Pt(0,  5))
# intersect = check_intersect(l1, l2)

# print(intersect[0]*l1.u + np.array([l1.P0.x, l1.P0.y]))
# print(intersect[1]*l2.u + np.array([l2.P0.x, l2.P0.y]))

lidar = Lidar(Pt(8, 5))
lidar.emit()

intersect_list = [None]*len(lidar.beam_list)
for i, b in enumerate(lidar.beam_list):
    best_d = np.inf
    for ml in map.line_list:
        intersect = check_intersect(ml, b)
        if intersect is not None and (intersect[1] < best_d and intersect[1] > 0. and abs(intersect[1]) <= b.L) and (intersect[0] > 0. and abs(intersect[0]) <= ml.L):
            best_d = intersect[1]
        
    if best_d is not np.inf:    
        
        intersect_list[i] = best_d*b.u + np.array([b.P0.x, b.P0.y])
        print(best_d, intersect_list[i])

ax = plt.subplot(1, 1, 1)

for line in map.line_list:
    ax.plot([line.pt1.x, line.pt2.x],
            [line.pt1.y, line.pt2.y], '-b')

for line in lidar.beam_list:
    ax.plot([line.pt1.x, line.pt2.x],
            [line.pt1.y, line.pt2.y], '--r')

for pt in intersect_list:
    if pt is not None:
        ax.plot(pt[0], pt[1], 'xg')

ax.set_aspect('equal', adjustable='box')
plt.show()


