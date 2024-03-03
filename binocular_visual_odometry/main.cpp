#include "util.h"
#include "stereo_vo.h"

int main(int argc, char **argv){

    if (argc != 2){
        std::cout << "[error] input data root" << std::endl;
        return 1;
    }

    //define file path
    std::string data_root    = std::string("../") + argv[1];
    std::string K_path       = data_root + "/calib.txt";
    std::string L_img_dir    = data_root + "/image_l";
    std::string R_img_dir    = data_root + "/image_r";
    std::string gt_pose_path = data_root + "/poses.txt";

    //load intrinsic and projection matrix
    std::vector<cv::Mat> Ks, Ps;
    get_K_mat(K_path, " ", Ks, Ps);

    std::cout << "L projection matrix: \n" << std::endl;
    std::cout << Ps[0] << std::endl;
    std::cout << "R projection matrix: \n" << std::endl;
    std::cout << Ps[1] << std::endl;

    //load ground truth poses
    std::vector<cv::Mat> gt_poses;
    load_gt_poses(gt_pose_path, " ", gt_poses);

    //load images
    std::vector<cv::Mat> L_imgs, R_imgs;
    load_img(L_img_dir, L_imgs);
    load_img(R_img_dir, R_imgs);

    //initilise pose history
    std::vector<cv::Mat> est_pose_hist;

    //initialise current pose
    cv::Mat c_pose;

    //initialise last tracking points on the left image
    std::vector<cv::Point2f> last_L_kps;

    //initialise last disparity map based on stereo matching
    cv::Mat last_disparity_map;

    //start mono vo
    for (int i = 0; i < R_imgs.size(); i ++){
        if (i == 0){
            compute_disparity_map(L_imgs[i], R_imgs[i], last_disparity_map);

            c_pose = gt_poses[i].clone();

            if (c_pose.rows == 3){
                cv::Mat tmp = (cv::Mat_<double>(1, 4) << 0., 0., 0., 1.);
                c_pose.push_back(tmp);
            }
        }
        else{
            //start mono vo
            //is_display, is_track, detect_mode
            cv::Mat R, t;

            stereo_vo(L_imgs[i - 1], 
                      R_imgs[i - 1],
                      L_imgs[i],
                      R_imgs[i], 
                      Ps, 
                      true, 
                      0, 
                      R, 
                      t, 
                      last_disparity_map,
                      last_L_kps); 

            cv::Mat T12;
            //T21 => transform normalised coordinate at frame {2} to frame {1}
            get_T_mat(R, t, T12);

            //transform back to initial frame
            //c_pose => transform normalised coordinate at frame {1} to frame {initial}
            c_pose = c_pose*T12;
        }

        //store estimation history
        est_pose_hist.push_back(c_pose.clone());
    }

    save_poses_to_csv("pose.csv", est_pose_hist);
}
