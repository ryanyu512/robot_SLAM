from sim import *

sim_params = {'dt': 0.02}

robot_params = {'init_x': 3.,
                'init_y': 8.,
                'init_yaw': 0.,
                'init_v': 0.,
                'init_theta_dot': 0.,
                'pos_std': 0.1,
                'theta_std': np.deg2rad(2.)}

SLAM_params = {'update_distance': 1.0,
               'update_ang': np.deg2rad(20.),
               'update_weight': [1, 1, 1]}

sim(sim_params,
    robot_params,
    SLAM_params)

