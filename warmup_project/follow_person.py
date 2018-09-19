#!/usr/bin/env python

from __future__ import print_function
from geometry_msgs.msg import PointStamped, PointStamped, Twist
from std_msgs.msg import Header
from neato_node.msg import Bump
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
from datetime import datetime
import statistics
import time, numpy, math, rospy

class FollowPerson(object):
    """follows a person"""

    def __init__(self):
        rospy.init_node("FollowPerson")
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.process_scan)
        self.stop = self.make_twist(0,0)
        self.xs = None
        self.ys = None
        self.xsf = None
        self.ysf = None

    def make_twist(self, x, theta):
        """ Takes x and angular velocity and creates the appropriate twist
        to publish to the cmd_vel topic."""
        send = Twist()
        send.linear.x = x
        send.linear.y = 0
        send.linear.z = 0
        send.angular.x = 0
        send.angular.y = 0
        send.angular.z = theta
        return send

    def show_plot(self):
        plt.plot(self.xs, self.ys, 'ro')
        plt.plot(0,0, 'bo', markersize=15)
        #plt.show()

    def process_scan(self, m):
        ranges = m.ranges
        view_angle = 50
        infront = ranges[0:int(view_angle/2)]+ranges[int(360-view_angle/2):360]
        xs = []
        ys = []
        xsf = []
        ysf = []
        for i in range(len(ranges)):
            if i<len(infront):
                if infront[i] !=0 and infront[i] < 1.5:
                    if i >= view_angle/2:
                        theta = math.radians(90-(view_angle-i))
                    else:
                        theta = math.radians(i+90)
                    r = infront[i]
                    xf = math.cos(theta)*r
                    yf = math.sin(theta)*r
                    #theta = i+90
                    #if theta<90+view_angle/2 and theta>90-view_angle/2 and r<1.5 and r!=0:
                    xsf.append(xf)
                    ysf.append(yf)

            if ranges[i] != 0:
                theta = math.radians(i+90)
                r = ranges[i]
                x = math.cos(theta)*r
                y = math.sin(theta)*r
                xs.append(x)
                ys.append(y)

        self.xs = xs
        self.ys = ys
        self.xsf = xsf
        self.ysf = ysf

    def get_points_ahead(self, total_angle):
        x = self.xs_infront
        y = self.ys_infront

        x_list = x[0:int(total_angle/2)]+x[int(360-total_angle/2):360]
        y_list = y[0:int(total_angle/2)]+y[int(360-total_angle/2):360]
        plt.plot(x_list, y_list, 'yo', markersize=10)
        return x_list, y_list

    def center_of_mass(self, x, y):
        x_cord = sum(x)/len(x)
        y_cord = sum(y)/len(y)
        self.show_plot()
        plt.plot(x_cord, y_cord, 'go', markersize=15)
        return (x_cord, y_cord)

    def cart_to_polar(self, x, y):
        r = math.sqrt(y**2 + x**2)
        # subtract 90 to account for robot forward heading=90
        theta = math.degrees(numpy.arctan2(y,x))-90
        return (r, theta)

    def drive_to_target(self, r, t):
        goal_d = 0.5    # desired distance away from target
        goal_t = 0      # desired angle away from goal, 0 to face target

        err_d = r - goal_d  # error terms
        err_t = t - goal_t

        kp_d = 0.5      # proportional control constants
        kp_t = 0.025

        x_vel = kp_d*err_d
        t_vel = kp_t*err_t

        send = self.make_twist(x_vel, t_vel)
        self.pub.publish(send)

    def run(self):
        while True:
            if isinstance(self.xsf, list):
                t_x, t_y = self.center_of_mass(self.xsf, self.ysf)
                plt.plot(self.xsf, self.ysf, 'yo', markersize=10)
                plt.plot(t_x, t_y)
                #plt.plot(self.xsf, self.ysf, 'yo', markersize=10)
                r, theta = self.cart_to_polar(t_x, t_y)
                print(r, theta)
                self.drive_to_target(r, theta)
                # plt.show()

if __name__ == '__main__':
    node = FollowPerson()
    node.run()
