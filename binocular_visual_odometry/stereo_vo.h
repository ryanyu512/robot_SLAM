#ifndef STEREO_VO_H
#define STEREO_VO_H

#include <random>
#include <limits>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv4/opencv2/opencv.hpp>

void get_T_mat(const cv::Mat &R, 
               const cv::Mat &t, 
               cv::Mat &T_mat);

void detect_kpts(const cv::Mat &img, 
                 const int mode,
                 std::vector<cv::KeyPoint> &kp);

void get_tracking_matches(const cv::Mat &img1,
                          const cv::Mat &img2,
                          const bool is_display,
                          const int  detect_mode,
                          std::vector<cv::Point2f> &last_kps,
                          std::vector<cv::Point2f> &gd_kps1,
                          std::vector<cv::Point2f> &gd_kps2);

void compute_disparity_map(const cv::Mat &L_img, 
                           const cv::Mat &R_img, 
                           cv::Mat &disparity_map);

void stereo_vo(const cv::Mat &pL_img,
               const cv::Mat &pR_img,
               const cv::Mat &cL_img,
               const cv::Mat &cR_img,
               const std::vector<cv::Mat> &Ps,
               const bool is_display,
               const int detect_mode,
               cv::Mat &R, 
               cv::Mat &t,
               cv::Mat &last_disparity_map,
               std::vector<cv::Point2f> &last_L_kps);

#endif
