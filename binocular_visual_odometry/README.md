Two folded approaches (feature matching or tracking approach) are implemented to achieve monocular visual odometry. For the first approach, key features are detected by the GFTT detector and matched by FLANN while KLT optical flow is used to track good matches in the second approach. Then, essential matrix is constructed based on good matches. Finally, rotation matrix and unit translation vector are recovered. Due to scale ambiguity, translation vector norm is extracted from the ground truth to recover the actual translation for correctness verification.

1. main.cpp: used for demo
2. util.h/cpp: define frequently used functions (i.e. loading camera parameters)
3. mono_vo.h/cpp: define feature extraction and pose estimation functions
4. visualise_results.py: used for visualise and check if the estimation result is reasonable
