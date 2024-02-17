import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from util import *

class Node():
    def __init__(self, id, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.id = id

    def connect(self, n, meas):
        self.neighbor.append([n, meas])

class Edge():
    def __init__(self, vi_ind, vj_ind, meas, w):
        self.vi_ind = vi_ind
        self.vj_ind = vj_ind
        self.meas = meas
        self.w    = w

    def get_info(self):
        
        return self.vi_ind, self.vj_ind, self.meas, self.w

class Graph():
    def __init__(self):
        self.node_list = {}
        self.edge_list = []
        self.N_node = 0
        self.N_edge = 0

    def add_edge(self, e):
        self.edge_list.append(e)
        self.N_edge += 1

    def add_node(self, n):
        if n.id not in self.node_list:
            self.node_list[n.id] = [n, self.N_node]
            self.N_node += 1

    def get_node(self, nid):
        if nid in self.node_list:
            return self.node_list[nid]
    
    def update_node(self, n):
        if n.id in self.node_list:
            _, index = self.node_list[n.id]
            self.node_list[n.id] = [n, index]

    @staticmethod
    def compute_nz_constants(vi, vj, z_ij):
        xi, yi, thetai = vi.x, vi.y, vi.theta
        xj, yj, thetaj = vj.x, vj.y, vj.theta    

        #i frame relative to initial frame
        T_vi = compute_2d_tranfM(xi, yi, thetai)
        #j frame relative to initial frame
        T_vj = compute_2d_tranfM(xj, yj, thetaj)
        #estimate pose j relative to i based on measurement (z)
        T_zij = compute_2d_tranfM(z_ij[0], z_ij[1], z_ij[2])
        #estimate pose j relative to i based on estimation
        T_eij = np.matmul(np.linalg.inv(T_vi), T_vj)

        #compute non-zero terms in Hessian Matrix
        s = np.sin(thetai + z_ij[2])
        c = np.cos(thetai + z_ij[2])
        A = np.array([[-c, -s, s*(xi - xj) + c*(yj - yi)],
                      [ s, -c, c*(xi - xj) + s*(yi - yj)],
                      [ 0,  0,                        -1]])
        B = np.array([[ c, s, 0],
                      [-s, c, 0],
                      [ 0, 0, 1]])
        
        #compute difference between estimated relative pose and measured relative pose
        T_diff = np.matmul(np.linalg.inv(T_zij), T_eij)

        e = np.array([T_diff[0,2], 
                      T_diff[1,2], 
                      np.arctan2(T_diff[1, 0], T_diff[0, 0])])

        return e, A, B


class pose_graph():

    def __init__(self):
        self.graph = Graph()

    def load_g2o(self, file_path):

        # IXX IXY IXT IYY IYT ITT
        #   6   7   8   9  10  11
        #   0   1   2   3   4   5
        g2o = [0, 1, 2, 1, 3, 4, 2, 4, 5]
        with open(file_path, "r") as f:
            
            for line in f:
                if line.startswith("#"):
                    continue
                
                txts = line.split()
                if txts[0] == "VERTEX_SE2":
                    
                    pose = np.array([float(x) for x in txts[2:5]])
                    nid  = int(txts[1])
                    n    = Node(nid, pose[0], pose[1], pose[2])

                    self.graph.add_node(n)

                elif txts[0] == 'EDGE_SE2':
                    n1_id, n2_id = int(txts[1]), int(txts[2])
                    
                    meas = np.array([float(x) for x in txts[3:6]])
                    w = np.array([float(x) for x in txts[6:12]])
                    w = w[g2o].reshape((3, 3))

                    e = Edge(n1_id, n2_id, meas, w)
                    self.graph.add_edge(e)

    def load_toro(self, file_path):
        # IXX IXY IYY ITT IXT IYT
        #   6   7   8   9  10  11
        #   0   1   2   3   4   5
        toro = [0, 1, 4, 1, 2, 5, 4, 5, 3]

        with open(file_path, "r") as f:
            
            for line in f:
                if line.startswith("#"):
                    continue
                
                txts = line.split()
                if txts[0] == "VERTEX2":
                    
                    pose = np.array([float(x) for x in txts[2:5]])
                    nid  = int(txts[1])
                    n    = Node(nid, pose[0], pose[1], pose[2])

                    self.graph.add_node(n)

                elif txts[0] == 'EDGE2':
                    n1_id, n2_id = int(txts[1]), int(txts[2])
                    
                    meas = np.array([float(x) for x in txts[3:6]])
                    w = np.array([float(x) for x in txts[6:12]])
                    w = w[toro].reshape((3, 3))

                    e = Edge(n1_id, n2_id, meas, w)
                    self.graph.add_edge(e)

    def solve(self):
        t0 = time.time()
        e_total = 0.

        #initialise hessian matrix
        H = np.zeros((self.graph.N_node * 3, self.graph.N_node * 3))
        #initialise constraints 
        b = np.zeros((self.graph.N_node * 3, 1))
        for edge in self.graph.edge_list:
            
            #get edge info
            vi_id, vj_id, z_ij, w = edge.get_info()
            #load nodes
            [vi, vi_index], [vj, vj_index] = self.graph.get_node(vi_id), self.graph.get_node(vj_id)
            #A refers to pose_i, B refers to pose_j
            e, A, B = self.graph.compute_nz_constants(vi, vj, z_ij)

            et = np.transpose(e)
            tmp1 = np.matmul(et, w)
            tmp2 = np.matmul(np.transpose(A), w)
            tmp3 = np.matmul(np.transpose(B), w)
            
            bi  = np.matmul(tmp1, A)
            bj  = np.matmul(tmp1, B)
            Hii = np.matmul(tmp2, A)
            Hij = np.matmul(tmp2, B)
            Hjj = np.matmul(tmp3, B)

            # get the node index 
            i, j = vi_index, vj_index

            si, ei = i*3, (i + 1)*3
            sj, ej = j*3, (j + 1)*3

            H[si:ei, si:ei] += Hii
            H[si:ei, sj:ej] += Hij
            H[sj:ej, si:ei] += np.transpose(Hij)
            H[sj:ej, sj:ej] += Hjj

            b[si:ei, 0] += bi
            b[sj:ej, 0] += bj

            e_total += np.matmul(et, e)

        #anchor the first node
        #if we assume the delta of first node is always zero
        #based on the formulation of matrix A, then A = eye(3)
        H[0:3,0:3] += np.eye(3)

        #get compressed H
        H_ = sp.sparse.csr_matrix(H)
        dx = sp.sparse.linalg.spsolve(H_, -b)

        for k in list(self.graph.node_list.keys()):
            n, index = self.graph.get_node(k)

            i = index * 3
            n.x += dx[i]
            n.y += dx[i+1]
            n.theta += dx[i+2]
            n.theta = wrap_ang(n.theta)
            self.graph.update_node(n)
        dt = time.time() - t0

        print(f"used time: {dt*1e3 :.2f} ms, energy: {e_total:g}")

        return e_total

    def optimise(self, N_iteration = 10, is_plot = True):
        
        if is_plot:
            ax  = plt.subplot(1, 1, 1)

            o_node = []
            for k in list(self.graph.node_list.keys()):
                n, _ = pg.graph.get_node(k)
                o_node.append([n.x, n.y])
            o_node = np.array(o_node)

        e_prev = np.inf
        for _ in range(N_iteration):
            
            energy = self.solve()
            if energy >= e_prev:
                break

            e_prev = energy
            
        if is_plot:
            n_node = []
            for k in list(self.graph.node_list.keys()):
                n, _ = pg.graph.get_node(k)
                n_node.append([n.x, n.y])

            n_node = np.array(n_node)

            ax.plot(o_node[:, 0], o_node[:, 1], '.r')
            ax.plot(n_node[:, 0], n_node[:, 1], '.g')

            ax.legend(['raw data', 'optimised'])
            ax.set_aspect('equal', adjustable='box')
            plt.show()

pg = pose_graph()

# file_path = '/home/ryan/github_repository/robot_SLAM/pose_graph_slam/pg1.g2o'
file_path = '/home/ryan/github_repository/robot_SLAM/pose_graph_slam/killian-small.toro'

file_type = file_path.split('/')[-1].split('.')[-1]

if file_type == 'toro':
    print("=== load toro file ===")
    pg.load_toro(file_path)
elif file_type == 'g2o':
    print("=== load g2o file ===")
    pg.load_g2o(file_path)

if pg.graph.N_node > 0 and pg.graph.N_edge > 0:
    print("=== current graph ===")
    print(f"N_node: {pg.graph.N_node}")
    print(f"N_edge: {pg.graph.N_edge}")
    pg.optimise(is_plot = True)

