#!/usr/bin/env python

import numpy as np
import cv2
from relative_nav.msg import Keyframe, Match
import rospy
from cv_bridge import CvBridge, CvBridgeError

class RGBD_odom:
    def __init__(self):
        rospy.Subscriber("keyframe", Keyframe, self.kf_callback, queue_size=100)
        rospy.Subscriber("place_recognition", Match, self.match_callback, queue_size=100)
        self.bridge = CvBridge()

        self.keyframe_images = dict()

        if __name__ == '__main__':
            try:
                rospy.spin()
            except rospy.ROSInterruptException:
                pass

    def kf_callback(self, msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg.rgb, "mono16")
            depth = self.bridge.imgmsg_to_cv2(msg.rgb, "mono16")
        except CvBridgeError as e:
            print(e)

        id = str(msg.vehicle_id)+"_"+str(msg.keyframe_id)

        self.keyframe_images[id] = (rgb, depth)

    def match_callback(self, msg):
        from_id = msg.from_keyframe
        to_id = msg.to_keyframes[0]
        cv2.imshow('from', self.keyframe_images[from_id][0])
        cv2.imshow('to', self.keyframe_images[to_id][0])
        cv2.waitKey(1)



if __name__ == "__main__":
    rospy.init_node('simple_rgbd_odometry')
    thing = RGBD_odom()