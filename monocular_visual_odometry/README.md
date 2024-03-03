ORB detector couples with KLT optical flow to detect and track features. Based on stereo images, the disparity map is computed via StereoSGBM. Triangulation technique is used to recover the 3d points. To obtain the relative pose, Ceres library is chosen for minimisation of reprojection error since it could automatically compute the jacobian and hessian matrix to shorten development time.

1. main.cpp: used for demo
2. util.h/cpp: define frequently used functions (i.e. loading camera parameters)
3. stereo_vo.h/cpp: define feature extraction and pose estimation functions
4. visualise_results.py: used for visualise and check if the estimation result is reasonable
