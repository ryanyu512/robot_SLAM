#include "mono_vo.h"
#include "util.h"

void get_T_mat(const cv::Mat &R, 
               const cv::Mat &t, 
               cv::Mat &T_mat){

    cv::Mat tmp = (cv::Mat_<double>(4, 4) <<
             R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
             R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
             R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0),
                             0.,                 0.,                 0.,                 1.
             );
    T_mat = tmp.clone();

}

void get_matches(const cv::Mat &img1, 
                const cv::Mat &img2,
                const bool is_display,
                const bool is_track,
                const int detect_mode,
                std::vector<cv::KeyPoint> &kp1,
                std::vector<cv::KeyPoint> &kp2,
                std::vector<cv::DMatch> &g_ms){
    

    cv::Mat img_draw;

    if (is_track){

        cv::Mat g_img1, g_img2;
        cv::cvtColor(img1, g_img1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, g_img2, cv::COLOR_BGR2GRAY);

        compute_kpts(g_img1, detect_mode, kp1);

        std::vector<cv::Point2f> pt1(kp1.size()), pt2;
        std::vector<uchar> status;
        std::vector<float> error;
        for (int i = 0; i < pt1.size(); i ++)
            pt1[i] = kp1[i].pt;
        cv::calcOpticalFlowPyrLK(g_img1, g_img2, pt1, pt2, status, error);

        if (is_display){
            img_draw = img2.clone();
            for (int i = 0; i < pt2.size(); i++) {
                cv::Point2f d = pt2[i] - pt1[i];
                double n = std::sqrt(std::pow(d.x, 2) + std::pow(d.y, 2));
                if (status[i] and n <= 50) {
                    std::cout << bool(status[i]) << std::endl;
                    cv::circle(img_draw, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
                    cv::line(img_draw, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
                }
            }
        }

    }
    else{
        //initialise detector, descriptor and matcher
        cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
        cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        
        //detect key points
        compute_kpts(img1, detect_mode, kp1);
        compute_kpts(img2, detect_mode, kp2);

        //compute descriptor
        cv::Mat d1, d2;
        descriptor->compute(img1, kp1, d1);
        descriptor->compute(img2, kp2, d2);

        //compute matching
        std::vector<std::vector<cv::DMatch>> knn_ms;
        matcher.knnMatch(d1, d2, knn_ms, 2);

        //obtain good matches
        /*
        1. we keep two best two matches
        2. if m1.distance ~= m2.distance => two matches equally good => we don't know how to choose
        3. however, if m1.distance < 0.7 * m2.distance => we may guess that m1 match is much better
        */
        const float ratio_thres = 0.7f;
        for (int i = 0; i < knn_ms.size(); i++){
            if (knn_ms[i].size() >= 2 and knn_ms[i][0].distance < ratio_thres * knn_ms[i][1].distance){
                g_ms.push_back(knn_ms[i][0]);
            }
        }

        if (is_display){
            //draw matches
            cv::drawMatches( img1, kp1, img2, kp2, g_ms, img_draw, cv::Scalar::all(-1),
                            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        }
    }

    if (is_display){
        cv::imshow("draw", img_draw);
        cv::waitKey(100);
    }
}

void compute_kpts(const cv::Mat &img, 
                  const int mode,
                  std::vector<cv::KeyPoint> &kp){
    
    if (mode == 0){
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        detector->detect(img, kp);
    }
    else{
        cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20);
        detector->detect(img, kp);
    }
}

void compute_E(const cv::Mat &K, 
               const std::vector<cv::KeyPoint> &kpt1, 
               const std::vector<cv::KeyPoint> &kpt2,
               const std::vector<cv::DMatch> &g_ms,
               cv::Mat &E, 
               cv::Mat &R,
               cv::Mat &t){

  std::vector<cv::Point2d> gd_pts1(g_ms.size());
  std::vector<cv::Point2d> gd_pts2(g_ms.size());

  for (int i = 0; i < g_ms.size(); i++) {
    gd_pts1[i] = kpt1[g_ms[i].queryIdx].pt;
    gd_pts2[i] = kpt2[g_ms[i].trainIdx].pt;
  }

  //extract principal point
  cv::Point2d pp(K.at<double>(0, 2), K.at<double>(1, 2));  

  //extract focal length      
  double f = K.at<double>(0, 0);    

  //compute essential matrix        
  E = cv::findEssentialMat(gd_pts1, gd_pts2, f, pp);

  //recover pose and give the most possible R and t
  //t is only a directional vector
  cv::recoverPose(E, gd_pts1, gd_pts2, R, t, f, pp);
}

void mono_vo(const std::vector<cv::Mat> &imgs, 
             const std::vector<cv::Mat> &gt_poses,
             const cv::Mat &K, 
             const bool is_display,
             const bool is_track,
             const int detect_mode){
    //initialise current pose 
    cv::Mat c_pose;

    //initilise pose history
    std::vector<cv::Mat> est_pose_hist;

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

            double t_norm;
            cv::Mat actual_t = (gt_poses[i] - gt_poses[i - 1]).col(3);
            t_norm = std::pow(actual_t.at<double>(0), 2) + std::pow(actual_t.at<double>(1), 2) + std::pow(actual_t.at<double>(2), 2);
            t_norm = std::sqrt(t_norm);


            std::vector<cv::KeyPoint> kp1, kp2;
            std::vector<cv::DMatch> g_ms;

            //compute features matching
            get_matches(imgs[i - 1],
                            imgs[i], 
                            is_display,
                            is_track,
                            detect_mode,
                            kp1,
                            kp2,
                            g_ms);
            
            if (is_track == false){
                //compute essential matrix
                cv::Mat E, R, t, T12, T21;
                compute_E(K, kp1, kp2, g_ms, E, R, t);

                //T21 => transform normalised coordinate at frame {1} to frame {2}
                get_T_mat(R, t*t_norm, T21);
                //T12 => transform normalised coordinate at frame {2} to frame {1}
                T12 = T21.inv();

                //transform back to initial frame
                //c_pose => transform noralised coordinate at frame {1} to frame {initial}
                c_pose = c_pose*T12;
            }
        }

        //store estimation history
        est_pose_hist.push_back(c_pose.clone());
    }

    save_poses_to_csv("est_pose.csv", est_pose_hist);
}