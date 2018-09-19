#!/usr/bin/env python

from __future__ import print_function
from geometry_msgs.msg import PointStamped, PointStamped
from std_msgs.msg import Header
from neato_node.msg import Bump
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import rospy

class AddMarker(object):
    """This node publishes a marker to rviz"""

    def __init__(self):
        rospy.init_node("AddMarker")
        self.pub = rospy.Publisher('vizualization_message', Marker, queue_size=10)
        rospy.Subscriber('/bump', Bump, self.process_bump)
        rospy.Subscriber('/scan', LaserScan, self.process_scan)

    def process_scan(self, m):
        ranges = m.ranges


    def process_bump(self, m):
        pass
        #print(m.leftFront)

    def draw_marker(self):
        marker = Marker()
        marker.header.frame_id = "/odom"
        marker.type = marker.SPHERE
        marker.pose.position.x = 1
        marker.pose.position.y = 2
        marker.pose.position.z = 0
        marker.scale = [0.25, 0.25, 0.25]
        marker.color.a = 0.75

        self.pub.publish(marker)

    def run(self):
        r = rospy.Rate(2)

        while not rospy.is_shutdown():
            self.draw_marker()
            r.sleep()

if __name__ == '__main__':
    node = AddMarker()
    node.run()
