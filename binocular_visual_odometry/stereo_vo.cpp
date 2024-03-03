#include "util.h"
#include "stereo_vo.h"

/*
these parameters should be encapsulated into a class and configured by an external file
(TODO)
*/

//determine if redetection of keypoints is required for tracking
const int MIN_PTS_THRESHOLD  = 1000;
const float MAX_TRACKING_ERR = 7.f;

//define stereo matching parameters
const int BLOCK = 11;
const int MAX_DISPARITY =  50;
const int MIN_DISPARITY = -50;

//triangulation parameters
const double MAX_DEPTH = 500.;

//optimisation parameters
const int MAX_ITERATION = 20;
const int SAMPLE_SIZE   = 6;


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
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(1000);
        detector->detect(img, kp);
    }
    else{
        cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(1000, 0.01, 7);
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
        cv::calcOpticalFlowPyrLK(g_img1, g_img2, pt1, pt2, status, error, cv::Size(21, 21), 6);

        //filter out bad matches
        for (int i = 0; i < pt2.size(); i++) {
            if (pt2[i].x < 0 or 
                pt2[i].y < 0 or 
                pt2[i].x >= g_img2.cols or 
                pt2[i].y >= g_img2.rows or 
                error[i] >= MAX_TRACKING_ERR)
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
            cv::waitKey(20);
        }

        prev_kps = gd_kps2;
}

void compute_disparity_map(const cv::Mat &L_img, 
                           const cv::Mat &R_img, 
                           cv::Mat &disparity_map){

    //parameters are defined based on guidelines from the below link
    //https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
    cv::Ptr<cv::StereoSGBM> stereo_matcher = cv::StereoSGBM::create(0,
                                                                    32,
                                                                    BLOCK,
                                                                    BLOCK * BLOCK * 8,
                                                                    BLOCK * BLOCK * 32);
    stereo_matcher->compute(L_img, R_img, disparity_map);

    //divide by 16 to obtain actual disparity
    disparity_map.convertTo(disparity_map, CV_64F);
    disparity_map /= 16.f;
}

void access_disparity(const cv::Mat &disparity_map,
                      const cv::Point2f &kp,
                      double &d){
    int c = kp.x;
    int r = kp.y;
    d = disparity_map.at<double>(r, c);    
}

void compute_stereo_matching(const std::vector<cv::Point2f> &pL_kps,
                             const std::vector<cv::Point2f> &cL_kps, 
                             const cv::Mat &last_disparity_map,
                             const cv::Mat &disparity_map, 
                             std::vector<cv::Point2d> &matched_pL_kps,
                             std::vector<cv::Point2d> &matched_pR_kps,
                             std::vector<cv::Point2d> &matched_cL_kps,
                             std::vector<cv::Point2d> &matched_cR_kps){

    for (int i = 0; i < pL_kps.size(); i ++){

        double d1, d2;
        access_disparity(last_disparity_map, 
                         pL_kps[i],
                         d1);

        access_disparity(disparity_map, 
                         cL_kps[i],
                         d2);

        if ((double(d1) >= MIN_DISPARITY and double(d1) <= MAX_DISPARITY) and 
            (double(d2) >= MIN_DISPARITY and double(d2) <= MAX_DISPARITY))
        {
            cv::Point2d pR_kp(pL_kps[i].x - d1, pL_kps[i].y);
            cv::Point2d cR_kp(cL_kps[i].x - d2, cL_kps[i].y);
            matched_pL_kps.push_back(pL_kps[i]);
            matched_pR_kps.push_back(pR_kp);

            matched_cL_kps.push_back(cL_kps[i]);
            matched_cR_kps.push_back(cR_kp);
        }

    }

}

void recover_3d(const std::vector<cv::Mat> &Ps,
                const std::vector<cv::Point2d> &matched_pL_kps,    
                const std::vector<cv::Point2d> &matched_pR_kps, 
                const std::vector<cv::Point2d> &matched_cL_kps,
                const std::vector<cv::Point2d> &matched_cR_kps,
                cv::Mat &pL_pt3d, 
                cv::Mat &cL_pt3d){

    cv::Mat cL_pt4d, pL_pt4d;
    
    cv::triangulatePoints(Ps[0], Ps[1], matched_pL_kps, matched_pR_kps, pL_pt4d);
    cv::divide(pL_pt4d, cv::repeat(pL_pt4d.row(3), 4, 1), pL_pt4d);
    pL_pt3d = cv::Mat(pL_pt4d, cv::Rect(0, 0, pL_pt4d.cols, 3)).clone();
    pL_pt3d.convertTo(pL_pt3d, CV_64F);

    cv::triangulatePoints(Ps[0], Ps[1], matched_cL_kps, matched_cR_kps, cL_pt4d);
    cv::divide(cL_pt4d, cv::repeat(cL_pt4d.row(3), 4, 1), cL_pt4d);
    cL_pt3d = cv::Mat(cL_pt4d, cv::Rect(0, 0, cL_pt4d.cols, 3)).clone();
    cL_pt3d.convertTo(cL_pt3d, CV_64F);
}

//define optimisation cost here
struct REPROJECT_COST {
  REPROJECT_COST(double f, 
                 double px, double py, 
                 double pu, double pv, 
                 double cu, double cv,
                 double p_wx, double p_wy, double p_wz, 
                 double c_wx, double c_wy, double c_wz):
    _f(f), _px(px), _py(py), _pu(pu), _pv(pv), _cu(cu), _cv(cv),_p_wx(p_wx), _p_wy(p_wy), _p_wz(p_wz), _c_wx(c_wx), _c_wy(c_wy), _c_wz(c_wz) {
  }

  template<typename T>
  bool operator()(
    const T *const rT, 
    T *residual) const {
        
        T r_vec[3] = {rT[0], rT[1], rT[2]};
        T r_mat[9];

        //convert angle axis to rotation matrix
        ceres::AngleAxisToRotationMatrix(r_vec, r_mat);

        /* from current xyz to previous uv*/
        //compute camera coorindate
        T c_cam_x = r_mat[0]*T(_c_wx) + r_mat[3]*T(_c_wy) + r_mat[6]*T(_c_wz) + rT[3];
        T c_cam_y = r_mat[1]*T(_c_wx) + r_mat[4]*T(_c_wy) + r_mat[7]*T(_c_wz) + rT[4];
        T c_cam_z = r_mat[2]*T(_c_wx) + r_mat[5]*T(_c_wy) + r_mat[8]*T(_c_wz) + rT[5];

        //convert to pixel
        T pu_pred = (T(_f)*c_cam_x + T(_px)*c_cam_z)/c_cam_z; 
        T pv_pred = (T(_f)*c_cam_y + T(_py)*c_cam_z)/c_cam_z;

        /* from previous xyz to current uv*/
        //compute camera coorindate
        T r_tx    = -(r_mat[0]*rT[3] + r_mat[1]*rT[4] + r_mat[2]*rT[5]);
        T r_ty    = -(r_mat[3]*rT[3] + r_mat[4]*rT[4] + r_mat[5]*rT[5]);
        T r_tz    = -(r_mat[6]*rT[3] + r_mat[7]*rT[4] + r_mat[8]*rT[5]);
        T p_cam_x = r_mat[0]*T(_p_wx) + r_mat[1]*T(_p_wy) + r_mat[2]*T(_p_wz) + r_tx;
        T p_cam_y = r_mat[3]*T(_p_wx) + r_mat[4]*T(_p_wy) + r_mat[5]*T(_p_wz) + r_ty;
        T p_cam_z = r_mat[6]*T(_p_wx) + r_mat[7]*T(_p_wy) + r_mat[8]*T(_p_wz) + r_tz;

        //convert to pixel
        T cu_pred = (T(_f)*p_cam_x + T(_px)*p_cam_z)/p_cam_z; 
        T cv_pred = (T(_f)*p_cam_y + T(_py)*p_cam_z)/p_cam_z;

        //get residual
        residual[0] = T(_pu) - pu_pred;
        residual[1] = T(_pv) - pv_pred; 
        residual[2] = T(_cu) - cu_pred;
        residual[3] = T(_cv) - cv_pred;

    return true;
  }

  const double _f, _px, _py, _pu, _pv,_cu, _cv, _p_wx, _p_wy, _p_wz, _c_wx, _c_wy, _c_wz;
};

void compute_reproject_err(const cv::Mat &P,
                             const cv::Mat &R, 
                             const cv::Mat &t,
                             const std::vector<cv::Point2d> &p_uv,
                             const std::vector<cv::Point2d> &c_uv,
                             const cv::Mat &p_pts3d, 
                             const cv::Mat &c_pts3d,
                             double &e1,
                             double &e2){
    
    //get transformation matrix
    cv::Mat T12, T21; 
    get_T_mat(R, t, T12);
    T21 = T12.clone().inv();

    //initialise ones matrix for homogenous coordinate
    cv::Mat ones_3d = cv::Mat::ones(1, p_pts3d.cols, CV_64F);
    cv::Mat p_pts4d = p_pts3d.clone();
    cv::Mat c_pts4d = c_pts3d.clone();

    p_pts4d.push_back(ones_3d.clone());
    c_pts4d.push_back(ones_3d.clone());
    
    //from current xyz to previous uv
    //(3, N) = (3, 4)*(4, 4)*(4, N)
    cv::Mat p_uv_pred = P*T12*c_pts4d;

    //from previous xyz to current uv
    //(3, N) = (3, 4)*(4, 4)*(4, N)
    cv::Mat c_uv_pred = P*T21*p_pts4d; 

    //normalise uv
    cv::divide(p_uv_pred, cv::repeat(p_uv_pred.row(2), 3, 1), p_uv_pred);
    cv::divide(c_uv_pred, cv::repeat(c_uv_pred.row(2), 3, 1), c_uv_pred);

    //compute total error
    e1 = 0.;
    e2 = 0.;

    int cnt = 0;
    for (int i = 0; i < p_uv.size(); i ++){
        double dpu = p_uv_pred.at<double>(0, i) - p_uv[i].x;
        double dpv = p_uv_pred.at<double>(1, i) - p_uv[i].y;
        double dcu = c_uv_pred.at<double>(0, i) - c_uv[i].x;
        double dcv = c_uv_pred.at<double>(1, i) - c_uv[i].y;

        if (std::abs(p_pts3d.at<double>(2, i)) >= MAX_DEPTH or 
            std::abs(c_pts3d.at<double>(2, i)) >= MAX_DEPTH){
            continue;
        }

        cnt += 1;
        e1 += std::sqrt((std::pow(dpu, 2) + std::pow(dcu, 2)));
        e2 += std::sqrt((std::pow(dpv, 2) + std::pow(dcv, 2)));
    }
    
    e1 /= double(cnt*2.);
    e2 /= double(cnt*2.);
}


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
               std::vector<cv::Point2f> &last_L_kps){
    
    //compute matching based on previous LEFT image and current LEFT image
    std::vector<cv::Point2f> track_pL_kps, track_cL_kps;
    get_tracking_matches(pL_img,
                         cL_img,
                         is_display,
                         detect_mode,
                         last_L_kps,
                         track_pL_kps, 
                         track_cL_kps);            

    //compute disparity map
    cv::Mat disparity_map;
    compute_disparity_map(cL_img, cR_img, disparity_map);

    //compute matched points on the previous RIGHT image and current RIGHT image
    std::vector<cv::Point2d> matched_pL_kps, matched_cL_kps;
    std::vector<cv::Point2d> matched_pR_kps, matched_cR_kps;
    compute_stereo_matching(track_pL_kps, 
                            track_cL_kps,
                            last_disparity_map,
                            disparity_map,
                            matched_pL_kps,
                            matched_pR_kps, 
                            matched_cL_kps,
                            matched_cR_kps);

    //compute triangulation
    cv::Mat cL_pts3d, pL_pts3d;
    recover_3d(Ps,
               matched_pL_kps, 
               matched_pR_kps,
               matched_cL_kps,
               matched_cR_kps,
               pL_pts3d,
               cL_pts3d);

    //optimise
    /*
    The following is the overall pipeline

    matched_pL_kps <=> matched_cL_kps based on KLT optical flow
    matched_pL_kps <=> matched_pR_kps based on last_disparity_map
    matched_cL_kps <=> matched_cR_kps based on disparity_map

    matched_pL_kps + matched_pR_kps => pL_pt3d based on triangulation 
    matched_cL_kps + matched_cR_kps => cL_pt3d based on triangulation 

    cL_pt3d => P*T       => p_uv_est vs p_uv => error 1
    p_uv    => P*(inv_T) => c_uv_est vs c_uv => error 2

    minimise error 1 and 2 based on non-linear least square
    */

    ceres::Problem problem;

    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int> distr(0, matched_pL_kps.size() - 1);

    ceres::Solver::Options options;     
    options.linear_solver_type = ceres::DENSE_QR;  
    options.minimizer_progress_to_stdout = false; 
    options.max_num_iterations = 200;  
    ceres::Solver::Summary summary;                

    double min_e1 = std::numeric_limits<double>::infinity();
    double min_e2 = std::numeric_limits<double>::infinity();

    //initialise index for random shuffle
    std::vector<int> rand_ind(matched_pL_kps.size());
    for (int i = 0; i < rand_ind.size(); i ++)
        rand_ind[i] = i;

    for (int i = 0; i < MAX_ITERATION; i ++){

        //initialise initial solution
        double rT[6] = {0., 0., 0., 0., 0., 0.};

        //initialise sample count
        int sample_cnt = 0;
        
        //get random index
        std::shuffle(rand_ind.begin(), rand_ind.end(), generator);

        for (int j = 0; j < rand_ind.size(); j ++){

            int ind = rand_ind[j];

            double pu = matched_pL_kps[ind].x;
            double pv = matched_pL_kps[ind].y;
            double cu = matched_cL_kps[ind].x;
            double cv = matched_cL_kps[ind].y;

            double px = pL_pts3d.at<double>(0, ind);
            double py = pL_pts3d.at<double>(1, ind);
            double pz = pL_pts3d.at<double>(2, ind);
            double cx = cL_pts3d.at<double>(0, ind);
            double cy = cL_pts3d.at<double>(1, ind);
            double cz = cL_pts3d.at<double>(2, ind);

            //filter unrealistic trangulation result
            if (std::abs(pz) >= MAX_DEPTH or 
                std::abs(cz) >= MAX_DEPTH){
                continue;
            }

            problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<REPROJECT_COST, 4, 6>(
                new REPROJECT_COST(Ps[0].at<double>(0, 0), //focal length
                                   Ps[0].at<double>(0, 2), //px
                                   Ps[0].at<double>(1, 2), //py
                                   pu, pv,                 //uv on i - 1th image
                                   cu, cv,                 //uv on ith image
                                   px, py, pz,             //3d pts at i - 1 time
                                   cx, cy, cz)),           //3d pts at i time
                nullptr,            
                rT                  
            );

            sample_cnt += 1;

            if (sample_cnt >= SAMPLE_SIZE)
                break;
        }

        //get solution
        ceres::Solve(options, &problem, &summary);

        //update roatation matrix and translation vector
        double tmp[9];
        ceres::AngleAxisToRotationMatrix(rT, tmp);
        
        cv::Mat R_temp, t_temp;
        R_temp = (cv::Mat_<double>(3, 3) << tmp[0], tmp[3], tmp[6],
                                            tmp[1], tmp[4], tmp[7],
                                            tmp[2], tmp[5], tmp[8]);
        t_temp = (cv::Mat_<double>(3, 1) << rT[3], rT[4], rT[5]);
        
        //evaluate estimation result
        double e1, e2;
        compute_reproject_err(Ps[0],
                                R_temp, 
                                t_temp,
                                matched_pL_kps,
                                matched_cL_kps,
                                pL_pts3d, 
                                cL_pts3d,
                                e1, e2);

        if (e1 < min_e1 and e2 < min_e2){
            min_e1 = e1;
            min_e2 = e2;
            R = R_temp.clone();
            t = t_temp.clone();
        }
    }

    //update last tracking points on the left image
    last_L_kps = track_cL_kps;

    //update last disparity map
    last_disparity_map = disparity_map.clone();
}

