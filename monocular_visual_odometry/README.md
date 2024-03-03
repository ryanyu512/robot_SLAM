Compared with monocular VO, binocular VO can recover depth and hence translation vector directly. In this project, ORB detector is chosen for fast feature detection on the previous left image. Then, KLT optical flow is utilised to track the current left image. Based on the feature points on both images, disparity map is computed via StereoSGBM. Then, triangulation technique is used to recover the 3d points of previous and current frame. To obtain the relative pose, ceres library is chosen for non-linear optimisation of reprojection error since it could automatically compute of jacobian and hessian matrix to shorten development time.

1. main.cpp: used for demo
2. util.h/cpp: define frequently used functions (i.e. loading camera parameters)
3. stereo_vo.h/cpp: define feature extraction and pose estimation functions
4. visualise_results.py: used for visualise and check if the estimation result is reasonable
