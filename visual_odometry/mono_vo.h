#ifndef MONO_VO_H
#define MONO_VO_H

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
                          std::vector<cv::Point2f> &prev_kps,
                          std::vector<cv::Point2f> &gd_kps1,
                          std::vector<cv::Point2f> &gd_kps2);

void get_feature_matches(const cv::Mat &img1, 
                         const cv::Mat &img2,
                         const bool is_display,
                         const int detect_mode,
                         std::vector<cv::Point2f> &gd_kps1,
                         std::vector<cv::Point2f> &gd_kps2);

void compute_E(const cv::Mat &K, 
               const std::vector<cv::Point2f> &gd_kps1,
               const std::vector<cv::Point2f> &gd_kps2,
               cv::Mat &E, 
               cv::Mat &R,
               cv::Mat &t);

void mono_vo(const cv::Mat &p_img,
             const cv::Mat &c_img,
             const std::vector<cv::Mat> &gt_poses,
             const cv::Mat &K, 
             const bool is_display,
             const bool is_track,
             const int detect_mode, 
             cv::Mat &R, 
             cv::Mat &t,
             std::vector<cv::Point2f> &prev_gd_kps);

#endif