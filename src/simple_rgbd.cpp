#include "simple_rgbd/simple_rgbd.h"

using namespace cv;
using namespace gtsam;

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
  Mat camera_matrix, distortion_coefficients;
  fs["camera_matrix"] >> camera_matrix;
  fs["distortion_coefficients"] >> distortion_coefficients;

  K_ = Cal3_S2(camera_matrix.at<double>(0,0), camera_matrix.at<double>(1,1),
               camera_matrix.at<double>(0,1), camera_matrix.at<double>(0,2),
               camera_matrix.at<double>(1,2));
  pixelNoise_ = noiseModel::Isotropic::Sigma(2, 1.0); // one pixel in u and v

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
  odom_.setCameraMatrix(camera_matrix);
  odom_.setTransformType(rgbd::ICPOdometry::RIGID_BODY_MOTION);
  //  odom_.setMaxDepth(4.0f);
  //  odom_.setMinDepth(0.0f);
  odom_.setMaxDepthDiff(0.5f);
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

    // Use GTSAM to find transform between images
    NonlinearFactorGraph graph;
    Values initialEstimate;

    // Add a prior on pose x1. This indirectly specifies where the origin is.
    Rot3 Identity(1, 0, 0, 0, 1, 0, 0, 0, 1);
    Pose3 origin(Identity, Point3(0, 0, 0));
    noiseModel::Diagonal::shared_ptr srcNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.001), Vector3::Constant(0.001)).finished()); // 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
    graph.emplace_shared<PriorFactor<Pose3> >(Symbol('x', 0), origin, srcNoise); // add directly to graph
    initialEstimate.insert(Symbol('x', 0), origin);

    // Add a prior for the dst location
    noiseModel::Diagonal::shared_ptr dstNoise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.5), Vector3::Constant(0.2)).finished()); // 50cm std on x,y,z 0.2 rad on roll,pitch,yaw
    graph.emplace_shared<PriorFactor<Pose3> >(Symbol('x', 1), origin, dstNoise); // add directly to graph
    initialEstimate.insert(Symbol('x', 1), origin);


    // Add measurements
    SimpleCamera camera(origin, K_);
    noiseModel::Isotropic::shared_ptr pointNoise = noiseModel::Isotropic::Sigma(3, 0.1);
    for (int i = 0; i < good_matches.size(); i++)
    {
      KeyPoint src_kp = src_keypoints[good_matches[i].queryIdx];
      Point2 src_point(src_kp.pt.x, src_kp.pt.y);
      double src_depth_kp = src_depth.at<double>(src_kp.pt);
      Pose3 src_landmark = camera.backproject(src_point, src_depth_kp);

      // Put in a prior and measurement for this landmark
      graph.emplace_shared<PriorFactor<Point3>>(Symbol('l',i), src_landmark, pointNoise);
      graph.emplace_shared<BetweenFactor<Pose3, Point3>>(Symbol('x', 0), Symbol('l', i), src_landmark, pointNoise);
      initialEstimate.insert(Symbol('l', i), src_landmark);

      KeyPoint dst_kp = dst_keypoints[good_matches[i].queryIdx];
      Point2 dst_point(dst_kp.pt.x, dst_kp.pt.y);
      double dst_depth_kp = dst_depth.at<double>(dst_kp.pt);
      Pose3 dst_landmark(Identity, camera.backproject(dst_point, dst_depth_kp));

      // Add the measurement from the second image
      graph.emplace_shared<BetweenFactor<Point3>>(Symbol('x', 1), Symbol('l', i), dst_landmark, pointNoise);
    }

    // Optimize
    Values result = DoglegOptimizer(graph, initialEstimate).optimize();
    result.print("Final Result:\n");

    ROS_INFO_STREAM("initial error = " << graph.error(initialEstimate));
    ROS_INFO_STREAM("final error = " << graph.error(result));
  }







//  for(unsigned i = 0; i < matches.size(); i++)
//  {
//    if(matches[i][0].distance < 0.3* matches[i][1].distance)
//    {
//      src_matched.push_back(src_keypoints[matches[i][0].queryIdx]);
//      dst_matched.push_back(dst_keypoints[matches[i][0].trainIdx]);
//      filtered_matches.push_back(matches[i]);
//    }
//  }

//  if ((double)src_matched.size()/(double)src_keypoints.size() > 0.1)
//  {
//    ROS_INFO("found sufficiently matched features: %f", (double)src_matched.size()/(double)matches.size());
//    Mat matched_img;


//  }
//  else
//  {
//    ROS_INFO("insufficient matches, %f", (double)src_matched.size()/(double)src_keypoints.size());
//  }



//  // Perform visual odometry to get transform
//  Mat mask(src_rgb.rows, src_rgb.cols, CV_8UC1);

//  Mat T;
//  if (odom_.compute(src_rgb, src_depth, mask, dst_rgb, dst_depth, mask, T))
//  {
//    Mat R = Mat(T, Rect(0, 0, 3, 3));
//    Mat tran = Mat(T, Rect(3, 0, 1, 3));
//    ROS_WARN_STREAM("computed T = \n" << T);
//    ROS_INFO_STREAM("rotation R = \n" << R);
//    ROS_INFO_STREAM("translation = \n" << tran);
//    ROS_INFO_STREAM("\n\n");

//    if (show_matches_)
//    {
//      imshow("from", keyframes[from_id][0]);
//      imshow("to", keyframes[to_id][0]);
//      waitKey(2);
//    }
//  }
//  else
//  {
//    ROS_INFO_STREAM("No transform computed, T = \n" << T);
//  }



}


}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "simple_rgbd");

  simple_rgbd::RGBD_Odom thing;

  ros::spin();
}
