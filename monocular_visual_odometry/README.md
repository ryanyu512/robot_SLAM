
This project utilise monocular visual odometry to estimate the relative pose based on sequential images. Orb feature detector/optical flow is used for detecting/tracking key points and FLANN is used for determining good matches. Then, essential matrix is constructed based on good matches. Finally, rotation matrix and unit translation vector are recovered. Due to scale ambiguity, translation vector norm is extracted from the ground truth to recover the actual translation for verification.

1. main.cpp: used for demo
2. util.h/cpp: define frequently used functions (i.e. loading camera parameters)
3. mono_vo.h/cpp: define feature extraction and pose estimation functions
4. visualise_results.py: used for visualise and check if the estimation result is reasonable
