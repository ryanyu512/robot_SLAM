import numpy as np

def compute_2D_Tmat(x, y, ang):
    return np.array([[np.cos(ang), -np.sin(ang),  x],
                     [np.sin(ang),  np.cos(ang),  y],
                     [          0,            0,  1.]])

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
