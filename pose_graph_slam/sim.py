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

    #initialise map
    map = Map2D()
    map.gen_map()

    #initialise pose graph slam
    pg = pose_graph()

    #initialise plotting
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2)
    plt.ion()
    plt.show()

    init_pts = None
    p_pts = None
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

        robot.update(dt, 0)
        robot.measure(map)

        #update node
        x_noise =  np.random.normal(robot.x, pos_std)
        y_noise =  np.random.normal(robot.y, pos_std)
        yaw_noise = np.random.normal(robot.yaw, theta_std)

        cn    = Node(pg.graph.N_node, x_noise, y_noise, yaw_noise)

        #check if new node should be added
        if is_optimise:
            pn, _ = pg.graph.get_node(0)
            p_pts = init_pts

            pg.graph.add_node(cn)
            is_add_node = True
            pose_hist.append([robot.x, robot.y, robot.yaw])
        else:
            if pg.graph.N_node > 0:
                pn, _ = pg.graph.get_node(pg.graph.N_node - 1)
            is_add_node = False
            if len(list(pg.graph.node_list.keys())) == 0:
                pg.graph.add_node(cn)
                p_pts = robot.pts_B
                init_pts = robot.pts_B
                pose_hist.append([robot.x, robot.y, robot.yaw])

                print(f"=== {pg.graph.N_node} nodes added ===")
            else:
                d = np.sqrt((pn.x - cn.x)**2 + (pn.y - cn.y)**2)
                d_ang = wrap_ang(pn.theta - cn.theta)
                if d >= update_distance or d_ang >= update_ang:
                    pg.graph.add_node(cn)
                    is_add_node = True
                    pose_hist.append([robot.x, robot.y, robot.yaw])

                    print(f"=== {pg.graph.N_node} nodes added ===")

        alliged_pts = None
        if is_add_node or is_optimise:

            #rough transformation based on current node (cn) and previous node (pn)
            T_wj = compute_2D_Tmat(cn.x,     
                                   cn.y,
                                   cn.theta)
            
            T_wi = compute_2D_Tmat(pn.x, 
                                   pn.y, 
                                   pn.theta)

            T_iw  = np.linalg.inv(T_wi)
            T_ij  = np.matmul(T_iw, T_wj)

            H_pts_B = np.concatenate((robot.pts_B, np.ones((robot.pts_B.shape[0], 1))), axis = 1)
            pts_ij  = np.transpose(np.matmul(T_ij, np.transpose(H_pts_B)))[:, 0:2]

            #fine tune transformation
            transformation_history, alliged_pts = icp(p_pts, pts_ij, distance_threshold = 0.5, )

            #get the transformation matrix
            T = np.eye(3)
            for t in transformation_history:
                t = np.concatenate((t, np.array([[0, 0, 1]])), axis = 0)
                T = np.matmul(t, T)
            T = np.matmul(T, T_ij)

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

            if is_optimise:
                pg.optimise()

        ax1.clear()
        for line in map.line_list:
            ax1.plot([line.pt1.x, line.pt2.x],
                     [line.pt1.y, line.pt2.y], '-b')
            
        ax1.plot(robot.x, robot.y, 'og')
        ax1.plot([robot.x, robot.x + np.cos(robot.yaw)],
                 [robot.y, robot.y + np.sin(robot.yaw)], 
                '-g')
        
        ph = np.array(pose_hist)
        ax1.plot(ph[:,0], ph[:,1], 'xg')

        for k in list(pg.graph.node_list.keys()):
            n, _ = pg.graph.get_node(k)
            ax1.plot(n.x, n.y, '+r')

        if alliged_pts is not None:
            ax2.clear()
            ax2.plot(alliged_pts[:,0], alliged_pts[:,1], '.r')
            ax2.plot(robot.pts_B[:,0], robot.pts_B[:,1], '+y')
            ax2.plot(p_pts[:,0], p_pts[:,1], '.g')
            ax2.set_aspect('equal', adjustable='box')

            #update point cloud at frame i
            p_pts = copy.copy(robot.pts_B)

        ax1.plot(robot.pts_G[:, 0], robot.pts_G[:, 1], '.y')

        ax1.set_aspect('equal', adjustable='box')
        plt.draw()
        plt.pause(.0001)


