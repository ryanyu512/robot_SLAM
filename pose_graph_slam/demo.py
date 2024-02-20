from sim import *

sim_params = {'end_t': 10,
              'dt': 0.02,
              'is_animate': True,
              'is_save_gif': True}

robot_params = {'init_x': 3.,
                'init_y': 8.,
                'init_yaw': 0.,
                'init_v': 0.,
                'init_theta_dot': 0.,
                'pos_std': 0.1,
                'theta_std': np.deg2rad(3)}

SLAM_params = {'update_distance': 1.,
               'update_ang': np.deg2rad(20.),
               'update_weight': [1000, 1000, 100]}

sim(sim_params,
    robot_params,
    SLAM_params)