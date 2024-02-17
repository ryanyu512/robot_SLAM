import numpy as np

def compute_2d_tranfM(x, y, theta):

    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta),  np.cos(theta), y],
                     [            0,              0, 1]])

def wrap_ang(theta):

    if theta < - np.pi:
        theta += np.pi*2. 
    elif theta > np.pi:
        theta -= np.pi*2.

    return theta