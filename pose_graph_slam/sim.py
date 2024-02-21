import os
import sys
import fcntl
import select
import copy

from icp import *
from map import *
from robot import *
from pose_graph import *

#=== handle keyboard input blocking issue===#
fd = sys.stdin.fileno()
fl = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

def sim(sim_params, 
        robot_params,
        SLAM_params):
    
    #initialise simulation setting
    end_t = sim_params['end_t']
    dt    = sim_params['dt']
    sim_steps = int(end_t/dt)

    is_animate  = sim_params['is_animate']
    is_save_gif = sim_params['is_save_gif']

    #define robot setting
    init_x = robot_params['init_x']
    init_y = robot_params['init_y']
    init_v = robot_params['init_v']
    init_yaw = robot_params['init_yaw']
    init_theta_dot = robot_params['init_theta_dot']
    pos_std = robot_params['pos_std']
    theta_std = robot_params['theta_std']

    #initialise SLAM setting
    update_distance = SLAM_params['update_distance']
    update_ang = SLAM_params['update_ang']

    #initialise simulation step
    lidar_meas_steps = np.ceil(1/Lidar().meas_rate/dt).astype(int)

    #initialise robot 
    robot = Robot()
    robot.init_state(x = init_x, y = init_y, yaw = init_yaw, v = init_v, theta_dot = init_theta_dot)
    pose_hist = []
    pose_noise_hist = []

    #initialise map
    map = Map2D()
    map.gen_map()

    #initialise grid map
    gmap = Occupancy_Map()
    gmap_noise = Occupancy_Map()

    #initialise pose graph slam
    pg = pose_graph()

    #initialise plotting
    fig, ax = plt.subplots(nrows = 2, ncols=2)
    plt.ion()
    plt.show()

    p_pts = None

    #initialise pts_B_hist
    pts_B_hist = []

    #initialise pts_G (to form an occupancy map)
    pts_G = []
    while True:

        #=== read keyboard inputs ===#
        c = sys.stdin.read(1)
        is_optimise = False
        if c != "" and c != "\n":
            print(f"receive {c} input")
        if c == 'q':
            break
        elif c == 'w':
            robot.v += 0.2
        elif c == 'e':
            robot.v -= 0.2
        elif c == 's':
            robot.theta_dot += np.deg2rad(3.)
        elif c == 'd':
            robot.theta_dot -= np.deg2rad(3.)
        elif c == 'z':
            robot.v = 0.
            robot.theta_dot = 0.
        elif c == 'x':
            robot.v = 0.
        elif c == 'c':
            robot.theta_dot = 0.
        if c == 'g':
            if pg.graph.N_edge > 0:
                is_optimise = True

        #update state and measure
        robot.update(dt, 0)
        robot.measure(map)

        #estimate current pose (it is simplified here for the sake of SLAM concept verification)
        x_noise =  np.random.normal(robot.x, pos_std)
        y_noise =  np.random.normal(robot.y, pos_std)
        yaw_noise = np.random.normal(robot.yaw, theta_std)

        #convert robot.pts_B based on noisy estimation
        pts_G = transform_pts(compute_2d_tranfM(x_noise, 
                                                y_noise,
                                                yaw_noise), 
                             robot.pts_B)
        #update grid map
        for pt in pts_G:
            gmap.update_grid(pt[0], pt[1])

        #update node
        cn    = Node(pg.graph.N_node, x_noise, y_noise, yaw_noise)

        #=== frontend section ===#

        is_add_node = False
        if pg.graph.N_node == 0:
            pg.graph.add_node(cn)
            pts_B_hist.append(robot.pts_B)
            pose_hist.append([robot.x, robot.y, robot.yaw])
            pose_noise_hist.append([x_noise, y_noise, yaw_noise])
            is_add_node = True
            print(f"=== {pg.graph.N_node} nodes added ===")
        else:
            if is_optimise:
                pn, _ = pg.graph.get_node(0)
                p_pts = pts_B_hist[0] #for loop closing
            else:
                pn, _ = pg.graph.get_node(pg.graph.N_node - 1)
                p_pts = pts_B_hist[-1] #for forming edge

            d = np.sqrt((pn.x - cn.x)**2 + (pn.y - cn.y)**2)
            d_ang = abs(wrap_ang(pn.theta - cn.theta))
            if d >= update_distance or d_ang >= update_ang or is_optimise:
                pg.graph.add_node(cn)
                pts_B_hist.append(robot.pts_B)
                pose_hist.append([robot.x, robot.y, robot.yaw])
                pose_noise_hist.append([x_noise, y_noise, yaw_noise])
                is_add_node = True
                print(f"=== {pg.graph.N_node} nodes added ===")

        if is_add_node and pg.graph.N_node >= 2:

            #rough transformation based on current node (cn) and previous node (pn)
            T_wj = compute_2D_Tmat(cn.x,     
                                   cn.y,
                                   cn.theta)
            
            T_wi = compute_2D_Tmat(pn.x, 
                                   pn.y, 
                                   pn.theta)

            print(f'node est - dx: {cn.x - pn.x}, dy: {pn.y - cn.y}, dang: {np.rad2deg(wrap_ang(cn.theta - pn.theta))}')

            T_iw  = np.linalg.inv(T_wi)
            T_ij  = np.matmul(T_iw, T_wj)

            pts_ij = transform_pts(T_ij, robot.pts_B)

            #fine tune transformation
            transformation_history, alliged_pts = icp(p_pts, pts_ij, distance_threshold = 0.5, )

            #get the transformation matrix
            T = np.eye(3)
            for t in transformation_history:
                t = np.concatenate((t, np.array([[0, 0, 1]])), axis = 0)
                T = np.matmul(t, T)
            T = np.matmul(T, T_ij)

            print(f'est by lidar data - dx: {T[0, 2]}, dy: {T[1, 2]}, dang: {np.rad2deg(np.arctan2(T[1,0], T[0,0]))}')

            #update edge
            w = np.eye(3)
            w[0, 0] = SLAM_params['update_weight'][0]
            w[1, 1] = SLAM_params['update_weight'][1]
            w[2, 2] = SLAM_params['update_weight'][2]

            e  = Edge(pn.id, 
                      cn.id, 
                      np.array([T[0,2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])]),
                      w)
            pg.graph.add_edge(e)

        #=== backend section ===#
        if is_optimise:
            pg.optimise()
            pts_G = np.array([])
            s_nid = sorted(list(pg.graph.node_list.keys()))
            gmap.reset()
            for i, k in enumerate(s_nid):
                n, _ = pg.graph.get_node(k)

                T = compute_2D_Tmat(n.x, n.y, n.theta)
                sub_pts_G = transform_pts(T, pts_B_hist[i])
                
                pts_G = np.vstack([pts_G, sub_pts_G]) if pts_G.size else sub_pts_G

                for pt in sub_pts_G:
                    gmap.update_grid(pt[0], pt[1])

        #=== visualisation ===#

        ax[0,0].clear()
        for line in map.line_list:
            ax[0,0].plot([line.pt1.x, line.pt2.x],
                     [line.pt1.y, line.pt2.y], '-b')
        
        #show ground truth
        ax[0,0].plot(robot.x, robot.y, 'og')
        ax[0,0].plot([robot.x, robot.x + np.cos(robot.yaw)],
                     [robot.y, robot.y + np.sin(robot.yaw)], 
                     '-g')
        ax[0,0].plot(robot.pts_G[:, 0], robot.pts_G[:, 1], '.y')

        #show icp alignment result
        if is_add_node and pg.graph.N_node >= 2:
            ax[0,1].clear()

            ax[0,1].plot(alliged_pts[:,0], alliged_pts[:,1], 'xr')
            ax[0,1].plot(p_pts[:,0], p_pts[:,1], '.g')

        #show map result based on raw pose and raw lidar data
        if (is_add_node or is_optimise) and pg.graph.N_node >= 2:
            xn, yn, yawn = pose_noise_hist[-1]
            T = compute_2d_tranfM(xn, yn, yawn)
            T_pts = transform_pts(T, pts_B_hist[-1])
            ax[1, 0].plot(T_pts[:, 0], T_pts[:, 1], '.g')

        #show map result based on loop closing result
        if is_optimise:
            ax[1,1].clear()
            w_map = gmap.extract_w_map()
            if len(w_map):
                ax[1, 1].plot(w_map[:,0], w_map[:,1], '.g')

        ax[0,0].set_title('Simulation')
        ax[0,1].set_title('ICP alignment')
        ax[1,0].set_title('No optimisation')
        ax[1,1].set_title('With optimisation')

        for i in range(2):
            for j in range(2):
                ax[i,j].set_aspect('equal', adjustable='box')
                ax[i,j].set_xlabel('pos - x (m)')
                ax[i,j].set_ylabel('pos - y (m)')

        fig.tight_layout()
        plt.draw()
        plt.pause(.0001)


