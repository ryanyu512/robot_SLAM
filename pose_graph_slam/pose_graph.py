import numpy as np
import scipy as sp
import time

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
            self.node_list[n.id] = n
            self.N_node += 1

    def get_node(self, nid):
        if nid in self.node_list:
            return self.node_list[nid]
    
    def update_node(self, n):
        if n.id in self.node_list:
            self.node_list[n.id] = n

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
            vi, vj = self.graph.get_node(vi_id), self.graph.get_node(vj_id)
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

            i, j = vi_id - 1, vj_id - 1
            # print(f'i: {i}, j {j}')
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
            n = self.graph.node_list[k]
            i = (n.id - 1) * 3
            n.x += dx[i]
            n.y += dx[i+1]
            n.theta += dx[i+2]
            n.theta = wrap_ang(n.theta)
            self.graph.update_node(n)
        dt = time.time() - t0

        print(f"used time: {dt*1e3 :.2f} ms")

        return e_total

    def optimise(self, N_iteration = 10):
        

        e_prev = np.inf
        for _ in range(N_iteration):
            
            energy = self.solve()
            print(f'iteration: {_}, e_prev: {e_prev :.5f}, energy: {energy :.5f}')
            if energy >= e_prev:
                break
            e_prev = energy

pg = pose_graph()

file_path = '/home/ryan/github_repository/robot_SLAM/pose_graph_slam/pg1.g2o'
pg.load_g2o(file_path)
pg.optimise()