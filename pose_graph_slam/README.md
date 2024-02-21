
[demo.webm](https://github.com/ryanyu512/robot_SLAM/assets/19774686/a93d0161-646d-4025-9124-9a43da9102a7)

This project utilizes pose graph SLAM for map generation. The relative estimation (delta x, delta y and delta heading) between subsequent poses could be estimated by inertial sensors. Also, the surrounding enviornment is measured by lidar and can also be used for relative estimation between subsequent posees. Pose graph SLAM acts as an optimiser to minimise the difference between relative estimation by inertial sensors and lidar sensor. Therefore, a more accurate map and localisation could be obtained at the same time. 

1. demo.py: used for testing different parameters of pose graph SLAM.
2. sim.py: custom function for running simulation
3. icp.py: for fine-tuning the transformation between subsequent poses
4. robot.py: define the properties of 2D vehicle
5. map.py: define the simulation 2D map and occupancy map
6. pose_graph.py define the pose graph algorithm
7. sensor.py: define the lidar sensor
8. util.py: define commonly used functions
9. geometry.py: define node, edge and transformation matrix computation
   
