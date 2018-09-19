#!/usr/bin/env python

from __future__ import print_function
from geometry_msgs.msg import PointStamped, PointStamped
from std_msgs.msg import Header
from neato_node.msg import Bump
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Twist
import tty
import select
import sys
import termios
import datetime
import rospy

class Drive(object):
    """This node is to test the neato interface and get it moving"""

    def __init__(self):
        rospy.init_node("Drive")
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/bump', Bump, self.process_bump)
        # rospy.Subscriber('/scan', LaserScan, self.process_scan)

    def process_scan(self, m):
        pass
        # print(m.ranges[0])

    def process_bump(self, m):
        pass
        #print(m.leftFront)

    def make_twist(self, x, theta):
        """
        Takes x and angular velocity and creates the appropriate twist
        to publish to the cmd_vel topic.
        """
        sendy = Twist()
        sendy.linear.x = x
        sendy.linear.y = 0
        sendy.linear.z = 0
        sendy.angular.x = 0
        sendy.angular.y = 0
        sendy.angular.z = theta
        return sendy

    def move_dist(self, distance):
        """
        Takes a distance in meters and moves it forward. Works under the
        timing that 0.5 cmd_vel = 1 ft/s.
        """
        speed = 0.5
        m2ft = 0.3048
        dist_ft = distance/m2ft
        sec = dist_ft

        message = self.make_twist(0.5, 0)
        self.pub.publish(message)

        start = datetime.now()
        if datetime.now()-start > sec:
            stop_msg = make_twist(0,0)
            self.pub.publish(stop_msg)



    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def run(self):
        while not rospy.is_shutdown():
            key = None
            while key != '\x03':
                key = self.getKey()

                if key == 'w':
                    move = [1,0]
                if key == 's':
                    move = [-1,0]
                if key == 'p':
                    move = [0,0]
                if key == 'a':
                    move = [0,1]
                if key == 'd':
                    move = [0,-1]

                print(move)
                lin_vel = 1
                ang_vel = 1

                x_vel = move[0]*lin_vel
                theta_vel = move[1]*ang_vel

                sendy = Twist()
                sendy.linear.x = x_vel
                sendy.linear.y = 0
                sendy.linear.z = 0
                sendy.angular.x = 0
                sendy.angular.y = 0
                sendy.angular.z = theta_vel

                self.pub.publish(sendy)


if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)
    node = Drive()
    node.run()
