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


# sx, sy, sa = 3., 8., 0.
# poses1 = np.array([[sx + i*0.1, 
#                     sy, 
#                     sa] for i in range(100)])

# sx, sy, sa = poses1[-1]
# poses2 = np.array([[sx, 
#                     sy, 
#                     wrap_ang(sa + np.deg2rad(-1.)*i)] for i in range(90)])

# sx, sy, sa = poses2[-1]
# poses3 = np.array([[sx, 
#                     sy - i*0.1, 
#                     wrap_ang(sa)] for i in range(20)])

# sx, sy, sa = poses3[-1]
# poses4 = np.array([[sx, 
#                     sy, 
#                     wrap_ang(sa + np.deg2rad(-1.)*i)] for i in range(90)])

# sx, sy, sa = poses4[-1]
# poses5 = np.array([[sx - i*0.1, 
#                     sy, 
#                     sa] for i in range(50)])

# sx, sy, sa = poses5[-1]
# poses6 = np.array([[sx, 
#                     sy, 
#                     sa + np.deg2rad(1.)*i] for i in range(90)])

# sx, sy, sa = poses6[-1]
# poses7 = np.array([[sx, 
#                     sy - i*0.1, 
#                     sa] for i in range(50)])

# poses = np.concatenate((poses1, poses2, poses3, poses4, poses5, poses6, poses7), axis = 0)

# lidar = Lidar()
# pts_hist = []
# T_pts_hist = []
# T_hist = [np.eye(3)]
# for i in range(poses.shape[0]):
#     lidar.emit(poses[i][0], poses[i][1], poses[i][2])
#     pts = lidar.receive(map, poses[i][0], poses[i][1], poses[i][2])
#     pts_hist.append(pts)
#     if i >= 1:
#         pose_j = pts_hist[-1]
#         T_wj = compute_2D_Tmat( poses[i][0] + np.random.normal(0, 0.02),     
#                                 poses[i][1] + np.random.normal(0, 0.02),
#                                 poses[i][2] + np.random.normal(0, np.deg2rad(0.1)))
#         T_wi = compute_2D_Tmat(poses[i - 1][0] + np.random.normal(0, 0.02), 
#                                poses[i - 1][1] + np.random.normal(0, 0.02), 
#                                poses[i - 1][2] + np.random.normal(0, np.deg2rad(0.1)))
#         T_iw = np.linalg.inv(T_wi)
#         H_pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis = 1)
#         T_ij  = np.matmul(T_iw, T_wj)
#         pts_ij = np.transpose(np.matmul(T_ij, np.transpose(H_pts)))[:, 0:2]
#         transformation_history, aligned_points, e = icp(pts_hist[-2], pts_ij, distance_threshold = 0.2, )
#         # print(f"e: {e}")
#         T = np.eye(3)
#         for t in transformation_history:
#             t = np.concatenate((t, np.array([[0, 0, 1]])), axis = 0)
#             T = np.matmul(t, T)
#         T = np.matmul(T, T_ij)
#         # T = T_ij
#         T_hist.append(np.matmul(T_hist[-1], T))

#     H_pts  = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis = 1)
#     T_pts_hist.append(np.transpose(np.matmul(T_hist[-1], np.transpose(H_pts))))

# ax = plt.subplot(1, 1, 1)

# for line in map.line_list:
#     ax.plot([line.pt1.x, line.pt2.x],
#             [line.pt1.y, line.pt2.y], '-b')


# # # for line in lidar1.beam_list:
# # #     ax.plot(line.pt2.x, line.pt2.y, '.r')

# # # ax.plot([lidar.beam_list[0].pt1.x, lidar1.beam_list[0].pt2.x], 
# # #         [lidar.beam_list[0].pt1.y, lidar1.beam_list[0].pt2.y], '--r')
# # # ax.plot([lidar1.c.x, lidar1.c.x + np.cos(heading1)],
# # #         [lidar1.c.y, lidar1.c.y + np.sin(heading1)], '-y')
# for i in range(1, len(poses)):
#     ax.plot([poses[i-1][0], poses[i][0]], 
#             [poses[i-1][1], poses[i][1]], '-o')

# # for pts in pts_hist:
# #     ax.plot(pts[:, 0],
# #             pts[:, 1], 'xg')
# for pts in T_pts_hist:
#     ax.plot(pts[:, 0],
#             pts[:, 1], '.y')
# ax.set_aspect('equal', adjustable='box')
# plt.show()


