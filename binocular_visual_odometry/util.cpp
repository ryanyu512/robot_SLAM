#include "util.h"

void save_poses_to_csv(const std::string &file_path, 
                       const std::vector<cv::Mat> poses){

      std::ofstream csv;

      csv.open ("pose.csv");
      for (int i = 0; i < poses.size(); i ++){
        cv::Mat pose = poses[i].clone();
        pose = pose.reshape(0, 1);

        std::string input_txt = "";
        for (int j = 0; j < pose.cols; j ++){
            std::string str_val = std::to_string(pose.at<double>(0, j));
            input_txt += str_val;
            if (j < pose.cols - 1)
                input_txt += ",";
        }
        csv << input_txt << "\n";
      }
      csv.close();

}

void get_K_mat(const std::string file_path, 
               const std::string delimiter, 
               std::vector<cv::Mat> &Ks, 
               std::vector<cv::Mat> &Ps){

    std::ifstream read_file(file_path);

    std::string txt;

    while (std::getline(read_file, txt)){
        size_t pos = 0;

        std::vector<double> tmp;
        while (true) {
            std::string token;
            pos = txt.find(delimiter);
            token = txt.substr(0, pos);

            txt.erase(0, pos + delimiter.length());
            tmp.push_back(std::stod(token));

            if (pos == std::string::npos)
                break;
        };
        Ps.push_back(cv::Mat(1, tmp.size(), CV_64F, tmp.data()).clone());
    }



    for (int i = 0; i < Ps.size(); i ++){
        cv::Rect roi(0, 0, 3, 3);
        Ps[i] = (Ps[i].reshape(0, 3)).clone(); //col, row = (4, 3)
        Ks.push_back(cv::Mat(Ps[i], roi).clone()); 
    }
}

void load_path_list(const std::string dir, 
                    std::vector<std::string> &list){

    const std::filesystem::path path{dir};

    for (auto const& dir_entry : std::filesystem::directory_iterator{ path })
        list.push_back(dir_entry.path().string());

    return;
}

void load_img(const std::string img_dir, 
              std::vector<cv::Mat>& imgs){

    std::vector<std::string> img_paths;
    load_path_list(img_dir, img_paths);
    std::sort(img_paths.begin(), img_paths.end());
    for (int i = 0; i < img_paths.size(); i++){
        imgs.push_back(cv::imread(img_paths[i], cv::IMREAD_COLOR));
    }

    return;
}

void load_gt_poses(const std::string file_path,
                   const std::string delimiter, 
                   std::vector<cv::Mat> &gt_poses){

    std::ifstream read_file(file_path);

    std::string txt;

    while (std::getline(read_file, txt)){
        size_t pos = 0;

        std::vector<double> tmp;
        while (true) {
            std::string token;
            pos = txt.find(delimiter);
            token = txt.substr(0, pos);

            txt.erase(0, pos + delimiter.length());
            tmp.push_back(std::stod(token));

            if (pos == std::string::npos)
                break;
        };
        gt_poses.push_back(cv::Mat(1, tmp.size(), CV_64F, tmp.data()).clone());
    }

    for (int i = 0; i < gt_poses.size(); i ++){
        gt_poses[i] = gt_poses[i].reshape(0, 3).clone(); //col, row = (4, 3)
    }
}