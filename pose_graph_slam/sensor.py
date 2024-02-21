import numpy as np
from geometry import *



class Lidar():

    def __init__(self):
        self.min_range = 0.1
        self.max_range = 10.
        self.max_ang_range = np.deg2rad(360.)
        self.resol = np.deg2rad(1.)
        self.N_beams = int(self.max_ang_range/self.resol) + 1
        self.beam_list = [None]*self.N_beams
        self.meas_rate = 10
        
    def emit(self, cx, cy, heading):
        pos_emit_ang = np.linspace(0.             ,  self.max_ang_range/2., int(self.max_ang_range/self.resol/2.)+ 1)
        neg_emit_ang = np.linspace(0. - self.resol, -self.max_ang_range/2., int(self.max_ang_range/self.resol/2.))
        emit_ang = np.concatenate((pos_emit_ang, neg_emit_ang), axis = 0)
        theta = emit_ang + heading
        xs = np.cos(theta)*(self.max_range) - np.sin(theta)*0. + cx
        ys = np.sin(theta)*(self.max_range) + np.cos(theta)*0. + cy

        cnt = 0
        for x, y in zip(xs, ys):
            self.beam_list[cnt] = Line(Pt(cx, cy), Pt(x, y))
            cnt += 1

    def receive(self, map, cx, cy, heading):
        pts_G = []
        for i, b in enumerate(self.beam_list):
            best_d = np.inf
            for ml in map.line_list:
                
                #check if any intersection
                intersect = self.check_intersect(ml, b)

                if intersect is None:
                    if (b.P0.x == ml.P0.x or b.P0.y == ml.P0.y):
                        # beam is parallel to a wall => completely blocked => break
                        best_d = np.inf #ensure no incorrect sensing
                        break
                    #just parallel but not blocked by a wall => continue
                    continue

                is_within_range = (intersect[1] >= self.min_range and intersect[1] <= b.L)
                is_block = intersect[1] < best_d
                is_within_wall  = (intersect[0] >= 0. and intersect[0] <= ml.L) 

                if is_block and is_within_range and is_within_wall:
                    best_d = intersect[1]
                
            if best_d is not np.inf:    
                pts_G.append(best_d*b.u + np.array([b.P0.x, b.P0.y]))

        return self.T2bodyframe(cx, cy, heading, np.array(pts_G)), np.array(pts_G)

    def T2bodyframe(self, cx, cy, heading, pt_cloud_G):

        #compute body frame (b) relative to global frame (g)
        T_gb = compute_2D_Tmat(cx, cy, heading)
        #compute global frame (g) relative to body frame (b)
        T_bg = np.linalg.inv(T_gb)
        #convert to homogeneous coordinate
        H_pt_cloud_G = np.concatenate((pt_cloud_G, 
                                       np.ones((pt_cloud_G.shape[0], 1))),
                                       axis = 1)
        #convert point cloud (relative to g frame) to point cloud (relative to b frame)
        H_pt_cloud_B = np.transpose(np.matmul(T_bg, np.transpose(H_pt_cloud_G)))
        
        return H_pt_cloud_B[:,0:2]

    @staticmethod
    def check_intersect(line1, line2):
        A = np.array([[line1.u[0], -line2.u[0]], 
                      [line1.u[1], -line2.u[1]]])
        b = np.array([[line2.P0.x - line1.P0.x], 
                      [line2.P0.y - line1.P0.y]])
        
        if abs(np.linalg.det(A)) >= 1e-5: #cannot be exactly zero
            return np.matmul(np.linalg.inv(A), b)
        else:
            return 