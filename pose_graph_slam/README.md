
[demo.webm](https://github.com/ryanyu512/robot_SLAM/assets/19774686/a93d0161-646d-4025-9124-9a43da9102a7)

This project utilizes model predictive control (MPC) to achieve full control of an autonomous vehicle. Based on sliding windows of reference trajectories, model predictive control (MPC) computes optimised control inputs to ensure the control inputs are smooth and respects motion constraints. 

1. demo.py: used for testing different parameters of MPC
2. ref_gen.py: generate references for motion control
3. MPC.py: formulate MPC for AGV
4. robot.py: define the properties of 2d vehicle
5. robot_base.py: define the physical constants of 2d vechicle
6. sim.py: custom function for running simulation
