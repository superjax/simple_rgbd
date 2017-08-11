#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/features2d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <relative_nav/Keyframe.h>
#include <relative_nav/Match.h>
#include <map>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace simple_rgbd
{

class RGBD_Odom
{
public:
  RGBD_Odom();

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber match_sub_;
  ros::Subscriber kf_sub_;

  bool show_matches_;

  void match_callback(const relative_nav::MatchConstPtr &msg);
  void kf_callback(const relative_nav::KeyframeConstPtr &msg);

 Eigen::Matrix4d ransac(double num_iterations, Eigen::MatrixXd src_3d_points, Eigen::MatrixXd dst_3d_points);
 Eigen::Vector3d find_centroid(std::vector<Eigen::Vector3d> points);

  std::map <std::string, std::vector<cv::Mat>> keyframes;

  cv::rgbd::RgbdOdometry odom_;
  cv::Ptr<cv::FeatureDetector> detector_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;

  double match_ratio_;

  cv::Mat camera_matrix_, distortion_coefficients_;
};

}
