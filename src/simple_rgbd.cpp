#include "simple_rgbd/simple_rgbd.h"

using namespace cv;

namespace simple_rgbd
{

RGBD_Odom::RGBD_Odom() :
  pnh_("~"), nh_()
{
  match_sub_ = nh_.subscribe("place_recognition", 10, &RGBD_Odom::match_callback, this);
  kf_sub_ = nh_.subscribe("keyframe", 10, &RGBD_Odom::kf_callback, this);

  show_matches_ = pnh_.param<bool>("show_matches", true);

  // Retrieve Camera parameters
  std::string filename = "/home/superjax/xtion_1/ost.yaml";
  ROS_INFO("opening %s", filename.c_str());
  FileStorage fs(filename, FileStorage::READ);
  fs["camera_matrix"] >> camera_matrix_;
  fs["distortion_coefficients"] >> distortion_coefficients_;

  std::vector<int> iterCounts(4);
  iterCounts[0] = 7;
  iterCounts[1] = 7;
  iterCounts[2] = 7;
  iterCounts[3] = 10;

  std::vector<float> minGradMagnitudes(4);
  minGradMagnitudes[0] = 12;
  minGradMagnitudes[1] = 5;
  minGradMagnitudes[2] = 3;
  minGradMagnitudes[3] = 1;

  // Configure the odometry
//  odom_.setCameraMatrix(camera_matrix_);
//  odom_.setTransformType(rgbd::ICPOdometry::RIGID_BODY_MOTION);
  //  odom_.setMaxDepth(4.0f);
  //  odom_.setMinDepth(0.0f);
//  odom_.setMaxDepthDiff(0.5f);
  //  odom_.setMaxTranslation(0.15);
  //  odom_.setMaxRotation(15);
  //  odom_.setMinGradientMagnitudes(Mat(minGradMagnitudes));
  //  odom_.setIterationCounts(Mat(iterCounts));

  keyframes.clear();

  // Initialize the ORB detector
  detector_ = ORB::create();
  matcher_ = DescriptorMatcher::create("BruteForce-Hamming");
  match_ratio_ = 0.0001;
}

void RGBD_Odom::kf_callback(const relative_nav::KeyframeConstPtr &msg)
{
  ROS_INFO("Got keyframe");
  // Save the keyframe (rgb and depth) and the proper id to the cache
  std::string image_name = std::to_string(msg->vehicle_id) + "_" + std::to_string(msg->keyframe_id);
  cv_bridge::CvImagePtr rgb_image = cv_bridge::toCvCopy(msg->rgb, "mono8");
  cv_bridge::CvImagePtr depth_image = cv_bridge::toCvCopy(msg->depth, "32FC1");
  std::vector<Mat> rgb_and_depth = {rgb_image->image, depth_image->image};

  keyframes[image_name] = rgb_and_depth;
}

void RGBD_Odom::match_callback(const relative_nav::MatchConstPtr &msg)
{
  ROS_INFO("Got match from %s, to %s", msg->from_keyframe.c_str(), msg->to_keyframes[0].c_str());
  std::string from_id = msg->from_keyframe;
  std::string to_id = msg->to_keyframes[0];

  Mat src_rgb = keyframes[from_id][0];
  Mat src_depth = keyframes[from_id][1];
  Mat dst_rgb = keyframes[to_id][0];
  Mat dst_depth = keyframes[to_id][1];

  // Compute A set of features
  std::vector<KeyPoint> src_keypoints, dst_keypoints;
  Mat src_descriptors, dst_descriptors;
  detector_->detectAndCompute(src_rgb, noArray(), src_keypoints, src_descriptors);
  detector_->detectAndCompute(dst_rgb, noArray(), dst_keypoints, dst_descriptors);

  std::vector<DMatch> matches;
  matcher_->match(src_descriptors, dst_descriptors, matches);
  // Find the closest match
  double min_dist = 100;
  for (int i = 0; i < dst_descriptors.rows; i++)
  {
    min_dist = (matches[i].distance < min_dist) ? matches[i].distance : min_dist;
  }

  std::vector<DMatch> good_matches;

  for (int i = 0; i < dst_descriptors.rows; i++)
  {
    if (matches[i].distance <= 35.0)
    {
      good_matches.push_back(matches[i]);
    }
  }

  if (good_matches.size() > 25)
  {
    ROS_INFO("Found sufficient matches in images, %lu", good_matches.size());
    Mat matched_img;
    drawMatches(src_rgb, src_keypoints, dst_rgb, dst_keypoints, good_matches, matched_img);
    imshow("matches", matched_img);
    waitKey(1);

    std::vector<KeyPoint> src_kp_undistorted, dst_kp_undistorted;
    undistortPoints(src_keypoints, src_kp_undistorted, camera_matrix_, distortion_coefficients_);
    undistortPoints(dst_keypoints, dst_kp_undistorted, camera_matrix_, distortion_coefficients_);

    // Set up the ponts for RANSAC
    std::vector<Point3d> src_points_3d, dst_points_3d;
    src_points_3d.resize(good_matches.size(), 3);
    dst_points_3d.resize(good_matches.size(), 3);
    double cx = camera_matrix_.at<double>(0, 2);
    double cy = camera_matrix_.at<double>(1, 2);
    double fx = camera_matrix_.at<double>(0, 0);
    double fy = camera_matrix_.at<double>(1, 1);
    for (int i = 0; i < good_matches.size(); i++)
    {
      // Back project this point into the 3d space
      double u = src_kp_undistorted[good_matches[i].queryIdx].pt.x;
      double v = src_kp_undistorted[good_matches[i].queryIdx].pt.y;
      double z = src_depth.at<double>(u,v);

      src_points_3d(i, 0) = (u - cx)*z/fx;
      src_points_3d(i, 1) = (v - cy)*z/fy;
      src_points_3d(i, 2) = z;

      u = dst_kp_undistorted[good_matches[i].queryIdx].pt.x;
      v = dst_kp_undistorted[good_matches[i].queryIdx].pt.y;
      z = dst_depth.at<double>(u,v);

      dst_points_3d(i, 0) = (u - cx)*z/fx;
      dst_points_3d(i, 1) = (v - cy)*z/fy;
      dst_points_3d(i, 2) = z;
    }

    // Run RANSAC
    Mat R, T;
    int inliers;
    std::vector<int> inlier_list;
    Mat current_other;
    ransac(100, Mat(src_points_3d), Mat(dst_points_3d), R, T, inliers, inlier_list, current_other);


  }
}

void  RGBD_Odom::ransac(const int num_iterations,
                        const cv::Mat &reference,
                        const cv::Mat &current,
                        cv::Mat &final_solution_1,
                        cv::Mat &final_solution_2,
                        int *inliers,
                        std::vector<int> *inlier_list,
                        cv::Mat &current_other)
{
  //check dimensions:
  ROS_ASSERT(reference.rows == current.rows);
  ROS_ASSERT(reference.rows > 3);  //need more than three matching features

  double best_total_error = 99999999999999999999.9; //something high...
  int best_size = 3; //keeping track of most inliers
  std::vector<double> best_errors; //best vector showing the error
  std::vector<int> best_inliers; //initialize to three numbers...
  best_inliers.push_back(1);
  best_inliers.push_back(2);
  best_inliers.push_back(3);

  //containers for the best solution:
  cv::Mat best_sol_1;
  cv::Mat best_sol_2;

  //main loop for RANSAC:
  #pragma omp parallel for shared( best_total_error, best_size) //,best_errors,best_sol_1,best_sol_2
  for(int i = 0; i < num_iterations; i++)
  {
    //containers for the solution at each iteration
    cv::Mat solution_1;
    cv::Mat solution_2;
    std::vector<double> error_temp;  //container for the error
    std::vector<int> inliers_temp;   //container for the inliers
    double temp_err; //sum of the error at each iteration

    cv::Mat reference_sample,current_sample,cur_2d_sample; //the container for reference & current samples
    reference_sample = cv::Mat();
    current_sample = cv::Mat();
    cur_2d_sample = cv::Mat();

    std::vector<int> list;

    //sample
    list = ransac_model_->randomSample(min_sample_,0,(reference.rows-1));

    //load the samples in
    for(size_t i = 0; i< list.size(); i++)
    {
      reference_sample.push_back(reference.row(list[i]));
      current_sample.push_back(current.row(list[i]));
      if(ransac_type_ == R_3PT) //Add other pertinent 3pt algorithm names here
      {
        cur_2d_sample.push_back(current_other.row(list[i]));
      }
    }

    //find a solution using the sampled points
    if(ransac_model_->sampleSolution(reference_sample,current_sample,solution_1,solution_2))
    {
      //check the error against all the points
      if(ransac_type_ == R_3PT)
      {
        //3D points need 3D reference and 2D current
        temp_err = ransac_model_->computeError(reference,current_other,solution_1,error_temp,inliers_temp,solution_2);
      }
      else
      {
        temp_err = ransac_model_->computeError(reference,current,solution_1,error_temp,inliers_temp,solution_2);
      }
    }
    else
    {
      continue;
    }

    #pragma omp critical
    if((temp_err < best_total_error && (int)inliers_temp.size() >= best_size) || (int)inliers_temp.size() > best_size)
    {
      //check to see if it is the best so far or better than our minimum threshold
      //two conditions, (better error and same or greater # inliers) OR (greater # inliers)

      //Have an improved guess, replace with new information:
      best_size = (int)inliers_temp.size();
      best_total_error = temp_err;
      best_errors = error_temp;
      best_inliers = inliers_temp;
      best_sol_1 = solution_1.clone();
      best_sol_2 = solution_2.clone();
      /// Exit Criteria: ( \attention NOT VALID WHILE USING OMP TO PARALLELIZE LOOP!)
      /// p = 1 - (1-ratio^N)^M, where N is sample size, M is the current iteration we are on, ratio is the
      /// best inlier ratio so far.  Typical cutoff values for p are 0.99 or 0.999
    }
  }//End of main loop

  if(best_sol_1.empty())
  {
    //ROS_ERROR("No inliers found in the RANSAC Portion!!");
    return;
  }

  //For using all the inliers to generate a least squares solution:
  cv::Mat best_reference, best_current;
  for(int i = 0; i < (int)best_inliers.size(); i++)
  {
    best_reference.push_back(reference.row(best_inliers[i]));
    best_current.push_back(current.row(best_inliers[i]));
  }

  if(ransac_type_ != R_3PT)
  {
    //Extract the projection matrix:
    RANSAC8pt *ransac_2d = (RANSAC8pt*)ransac_model_; //cast to get to the trianglePoint function
    ransac_2d->computePfromE(best_sol_1,best_reference,best_current,best_sol_2);

//    std::cout << "Best Essential Matrix:" <<std::endl;
//    std::cout << best_sol_1 << std::endl;
//    std::cout << "Best Projection Matrix:" <<std::endl;
//    std::cout << best_sol_2 << std::endl;

    cv::Mat final_1, final_2; //matricies from the SVD;
    bool valid;
//    if(best_inliers.size() > (size_t)8)
//    {
//      valid = ransac_model_->sampleSolution(best_reference,best_current,final_1,final_2);
//      ransac_2d->computePfromE(final_1,best_reference,best_current,final_2);
//    }
//    else
      valid = false;

//    std::cout << "Combined 1 Essential Matrix:" <<std::endl;
//    std::cout << final_1 << std::endl;
//    std::cout << "Combined 1 Projection Matrix:" <<std::endl;
//    std::cout << final_2 << std::endl;


    if(valid)
    {
      //use the least squares solution:
      final_1.copyTo(final_solution_1);
      final_2.copyTo(final_solution_2);
    }
    else
    {
      //use the RANSAC solution:
      best_sol_1.copyTo(final_solution_1);
      best_sol_2.copyTo(final_solution_2);
    }
  }
  else
  {
    //use the 3pt solution:
    best_sol_1.copyTo(final_solution_1);
    best_sol_2.copyTo(final_solution_2);
  }

  *inliers = (int)best_inliers.size(); //report # inliers
  *inlier_list = best_inliers;

  //ROS_INFO_THROTTLE(1,"RANSAC Inliers = %d", *inliers); //advertise the # of inliers
}


Eigen::Vector3d RGBD_Odom::find_centroid(std::vector<Eigen::Vector3d> points)
{
  Eigen::Vector3d sum;
  sum.setZero();

  for (int i = 0; i < points.size(); i++)
  {
    sum += points[i];
  }

  Eigen::Vector3d mean = sum/(double)points.size();
  return mean;
}



}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "simple_rgbd");

  simple_rgbd::RGBD_Odom thing;

  ros::spin();
}
