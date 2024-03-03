#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv4/opencv2/opencv.hpp>

void save_poses_to_csv(const std::string &file_path, 
                       const std::vector<cv::Mat> poses);

void get_K_mat(const std::string file_path, 
               const std::string delimiter, 
               std::vector<cv::Mat> &Ks,
               std::vector<cv::Mat> &Ps);

void load_path_list(const std::string dir, 
                    std::vector<std::string> &list);

void load_img(const std::string img_dir, 
              std::vector<cv::Mat>& imgs);

void load_gt_poses(const std::string file_path,
                   const std::string delimiter, 
                   std::vector<cv::Mat> &gt_poses);

#endif