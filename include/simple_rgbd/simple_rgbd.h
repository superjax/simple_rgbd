#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/features2d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <relative_nav/Keyframe.h>
#include <relative_nav/Match.h>
#include <map>

#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Values.h>

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

  std::map <std::string, std::vector<cv::Mat>> keyframes;

  cv::rgbd::RgbdOdometry odom_;
  cv::Ptr<cv::FeatureDetector> detector_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;

  gtsam::Cal3_S2 K_;
  // Define the camera observation noise model
  gtsam::noiseModel::Isotropic::shared_ptr pixelNoise_;

  double match_ratio_;
};

}
