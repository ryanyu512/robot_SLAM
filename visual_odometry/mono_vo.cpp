#include "mono_vo.h"
#include "util.h"

//determine if redetection of keypoints is required for tracking
const int MIN_PTS_THRESHOLD = 300;

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

void detect_kpts(const cv::Mat &img, 
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

void get_tracking_matches(const cv::Mat &img1,
                          const cv::Mat &img2,
                          const bool is_display,
                          const int  detect_mode,
                          std::vector<cv::Point2f> &prev_kps,
                          std::vector<cv::Point2f> &gd_kps1,
                          std::vector<cv::Point2f> &gd_kps2){

        cv::Mat g_img1, g_img2;
        cv::cvtColor(img1, g_img1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, g_img2, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> kp1;
        std::vector<cv::Point2f> pt1, pt2;
        if (prev_kps.size() <= MIN_PTS_THRESHOLD){
            detect_kpts(g_img1, detect_mode, kp1);
            pt1 = std::vector<cv::Point2f>(kp1.size());
            for (int i = 0; i < pt1.size(); i ++)
                pt1[i] = kp1[i].pt;
        }
        else{
            pt1 = prev_kps;
        }
        std::vector<uchar> status;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(g_img1, g_img2, pt1, pt2, status, error, cv::Size(21, 21), 5);

        //filter out bad matches
        for (int i = 0; i < pt2.size(); i++) {
            if (pt2[i].x < 0 or pt2[i].y < 0)
                status[i] = 0;
            else if (status[i]){
                gd_kps1.push_back(pt1[i]);
                gd_kps2.push_back(pt2[i]);
            }
        }

        if (is_display){
            cv::Mat img_draw = img2.clone();
            for (int i = 0; i < gd_kps2.size(); i++) {
                cv::circle(img_draw, gd_kps2[i], 2, cv::Scalar(0, 250, 0), 2);
                cv::line(img_draw, gd_kps1[i], gd_kps2[i], cv::Scalar(0, 250, 0));
            }
            cv::imshow("draw", img_draw);
            cv::waitKey(100);
        }

        prev_kps = gd_kps2;

        std::cout << "last good key points size: " << prev_kps.size() << std::endl;
}

void get_feature_matches(const cv::Mat &img1, 
                         const cv::Mat &img2,
                         const bool is_display,
                         const int detect_mode,
                         std::vector<cv::Point2f> &gd_kps1,
                         std::vector<cv::Point2f> &gd_kps2){
    
    //initialise detector, descriptor and matcher
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    
    //detect key points
    std::vector<cv::KeyPoint> kp1, kp2;
    detect_kpts(img1, detect_mode, kp1);
    detect_kpts(img2, detect_mode, kp2);

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
    std::vector<cv::DMatch> g_ms;
    const float ratio_thres = 0.7f;
    for (int i = 0; i < knn_ms.size(); i++){
        if (knn_ms[i].size() >= 2 and knn_ms[i][0].distance < ratio_thres * knn_ms[i][1].distance){
            g_ms.push_back(knn_ms[i][0]);
        }
    }

    //get good keypoints
    gd_kps1 = std::vector<cv::Point2f>(g_ms.size());
    gd_kps2 = std::vector<cv::Point2f>(g_ms.size());

    for (int i = 0; i < g_ms.size(); i++) {
        gd_kps1[i] = kp1[g_ms[i].queryIdx].pt;
        gd_kps2[i] = kp2[g_ms[i].trainIdx].pt;
    }

    if (is_display){
        //draw matches
        cv::Mat img_draw;
        cv::drawMatches( img1, kp1, img2, kp2, g_ms, img_draw, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        cv::imshow("draw", img_draw);
        cv::waitKey(100);
    }
}


void compute_E(const cv::Mat &K, 
               const std::vector<cv::Point2f> &gd_kps1,
               const std::vector<cv::Point2f> &gd_kps2,
               cv::Mat &E, 
               cv::Mat &R,
               cv::Mat &t){

  //extract principal point
  cv::Point2f pp(K.at<double>(0, 2), K.at<double>(1, 2));  

  //extract focal length      
  double f = K.at<double>(0, 0);    

  //compute essential matrix        
  E = cv::findEssentialMat(gd_kps1, gd_kps2, f, pp);

  //recover pose and give the most possible R and t
  //t is only a directional vector
  cv::recoverPose(E, gd_kps1, gd_kps2, R, t, f, pp);
}

void mono_vo(const cv::Mat &p_img,
             const cv::Mat &c_img,
             const std::vector<cv::Mat> &gt_poses,
             const cv::Mat &K, 
             const bool is_display,
             const bool is_track,
             const int detect_mode, 
             cv::Mat &R, 
             cv::Mat &t,
             std::vector<cv::Point2f> &prev_gd_kps){

    //compute matching
    std::vector<cv::Point2f> gd_kps1, gd_kps2;
    if (is_track)
        get_tracking_matches(p_img,
                             c_img, 
                             is_display,
                             detect_mode,
                             prev_gd_kps,
                             gd_kps1,
                             gd_kps2);
    else
        get_feature_matches(p_img,
                            c_img, 
                            is_display,
                            detect_mode,
                            gd_kps1,
                            gd_kps2);

    //compute essential matrix
    cv::Mat E;
    compute_E(K, gd_kps1, gd_kps2, E, R, t);
}

