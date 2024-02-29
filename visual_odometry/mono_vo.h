#ifndef MONO_VO_H
#define MONO_VO_H

#include <opencv4/opencv2/opencv.hpp>

void get_T_mat(const cv::Mat &R, 
               const cv::Mat &t, 
               cv::Mat &T_mat);

void compute_kpts(const cv::Mat &img, 
                  const int mode,
                  std::vector<cv::KeyPoint> &kp);

void get_matches(const cv::Mat &img1, 
                     const cv::Mat &img2,
                     const bool is_display,
                     const bool is_track,
                     const int detect_mode,
                     std::vector<cv::KeyPoint> &kp1,
                     std::vector<cv::KeyPoint> &kp2,
                     std::vector<cv::DMatch> &g_ms);

void compute_E(const cv::Mat &K, 
               const std::vector<cv::KeyPoint> &kpt1, 
               const std::vector<cv::KeyPoint> &kpt2,
               const std::vector<cv::DMatch> &g_ms,
               cv::Mat &E, 
               cv::Mat &R,
               cv::Mat &t);

void mono_vo(const std::vector<cv::Mat> &imgs, 
             const std::vector<cv::Mat> &gt_poses,
             const cv::Mat &K, 
             const bool is_display,
             const bool is_track,
             const int detect_mode);

#endif