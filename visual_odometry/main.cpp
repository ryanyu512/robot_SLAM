#include "util.h"
#include "mono_vo.h"

int main(int argc, char **argv){

    if (argc != 2){
        std::cout << "[error] input data root" << std::endl;
        return 1;
    }

    //define file path
    std::string data_root    = std::string("../") + argv[1];
    std::string K_path       = data_root + "/calib.txt";
    std::string img_dir      = data_root + "/image_l";
    std::string gt_pose_path = data_root + "/poses.txt";

    //load intrinsic matrix
    std::vector<cv::Mat> Ks;
    get_K_mat(K_path, " ", Ks);

    //note: this project just utilise left camera
    // std::cout << "=== K ===" << std::endl;
    // std::cout << Ks[0] << std::endl;

    //load ground truth poses
    std::vector<cv::Mat> gt_poses;
    load_gt_poses(gt_pose_path, " ", gt_poses);

    //load images
    std::vector<cv::Mat> imgs;
    load_img(img_dir, imgs);

    //start mono vo
    //is_display, is_track, detect_mode
    mono_vo(imgs, gt_poses, Ks[0], true, false, 1); 
}