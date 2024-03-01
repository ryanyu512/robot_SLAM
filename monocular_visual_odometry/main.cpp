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

    //initilise pose history
    std::vector<cv::Mat> est_pose_hist;

    //start mono vo
    cv::Mat c_pose;

    //initialsie the last good tracking points
    std::vector<cv::Point2f> last_kps;
    
    for (int i = 0; i < imgs.size(); i ++){
        if (i == 0){
            c_pose = gt_poses[i].clone();

            if (c_pose.rows == 3){
                cv::Mat tmp = (cv::Mat_<double>(1, 4) << 0., 0., 0., 1.);
                c_pose.push_back(tmp);
            }
        }
        else{

            /*
            Note: 
            1. Monocular VO cannot recover actual translation vector directly
            2. To verify implementation, I directly utilise the ground truth to compute the translation vector norm
            3. To really compute actual translation vector norm, we can use inertial sensor or GPS
            */

            double t_norm = 0.;
            cv::Mat actual_t = (gt_poses[i] - gt_poses[i - 1]).col(3);
            for (int j = 0; j < 3; j ++)
                t_norm += std::pow(actual_t.at<double>(j), 2);
            t_norm = std::sqrt(t_norm);

            //start mono vo
            //is_display, is_track, detect_mode
            cv::Mat R, t;

            mono_vo(imgs[i - 1], imgs[i], Ks[0], true, true, 1, R, t, last_kps); 

            cv::Mat T12, T21;
            //T21 => transform normalised coordinate at frame {1} to frame {2}
            get_T_mat(R, t*t_norm, T21);
            //T12 => transform normalised coordinate at frame {2} to frame {1}
            T12 = T21.inv();

            //transform back to initial frame
            //c_pose => transform normalised coordinate at frame {1} to frame {initial}
            c_pose = c_pose*T12;
        }

        //store estimation history
        est_pose_hist.push_back(c_pose.clone());
    }

    save_poses_to_csv("pose.csv", est_pose_hist);
}