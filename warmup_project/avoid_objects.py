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

class AvoidObject(object):
    """avoids objects. Runs a potential field algorithm where every non-zero
       laser scan point is treated as a repulsive point."""

    def __init__(self):
        """initialize ROS stuff."""
        rospy.init_node("AvoidObject")
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.process_scan)
        rospy.Subscriber('/bump', Bump, self.process_bump)

        self.stop = self.make_twist(0,0)
        self.xs = None
        self.ys = None
        self.go = True

    def make_twist(self, x, theta):
        """takes x and angular velocity and creates the appropriate twist
        to publish to the cmd_vel topic."""
        send = Twist()
        send.linear.x = x
        send.linear.y = 0
        send.linear.z = 0
        send.angular.x = 0
        send.angular.y = 0
        send.angular.z = theta
        return send

    def process_scan(self, m):
        """process scan data to clean 0s and only look within a certain max
           distance from the laser scanner."""
        ranges = m.ranges
        xs = []
        ys = []
        xsf = []
        ysf = []

        for i in range(len(ranges)):
            if ranges[i] != 0 and ranges[i]<1.5:
                theta = math.radians(i+90)
                r = ranges[i]
                xf = math.cos(theta)*r
                yf = math.sin(theta)*r
                xs.append(xf)
                ys.append(yf)

        self.xs = xs
        self.ys = ys

    def process_bump(self, m):
        """uses the bump sensor as an e-stop for the neato."""
        lf = m.leftFront
        rf = m.rightSide

        if lf != 0 or rf != 0:
            self.pub.publish(self.stop)
            self.go = False

    def cart_to_polar(self, x, y):
        """descriptive title."""
        r = math.sqrt(y**2 + x**2)

        # subtract 90 to account for robot forward heading=90
        theta = math.degrees(numpy.arctan2(y,x))-90
        return (r, theta)

    def points_to_vector(self, x_list,y_list,a,radius):
        """Takes in a point cloud of obstacles and defines a vector at (0, 0)
           on the neato's position that points along the gradient of the
           potential field. Adds a positive vector that pushes the neato to
           move forward along its heading. "a" is a scaling factor. Returns the
           vector to which the neato should turn to align itself."""
        neg_vector = [0,0]
        s = 3 # spread of field
        for i in range(len(x_list)):
            x = x_list[i]
            y = y_list[i]
            dist = math.sqrt(y**2 + x**2)
            theta = math.atan2(y, x)
            x = -a * (s + radius - dist) * math.cos(theta)
            y = -a * (s + radius - dist) * math.sin(theta)
            neg_vector[0] = neg_vector[0] + x
            neg_vector[1] = neg_vector[1] + y
        pos_vector = [0, 0*a]
        vector_move = [pos_vector[0] + neg_vector[0], pos_vector[1] + neg_vector[1]];

        # try plotting some validation
        plt.plot(self.xs, self.ys, 'ro', markersize=10)     # all points
        plt.plot(0, 0, 'go', markersize=15)                 # neato at 0,0
        plt.quiver(vector_move[0], vector_move[1])     # resultant vector

        return vector_move

    def drive_to_target(self, r, t):
        """proportional control for driving a robot towards a goal point in
           polar coordinates. Experiemtnally tuned Kp values."""
        goal_d = 0      # desired distance away from target
        goal_t = 0      # desired angle away from goal, 0 to face target

        err_d = r - goal_d  # error terms
        err_t = t - goal_t

        kp_d = 0.005      # proportional control constants
        kp_t = 0.01

        x_vel = kp_d*err_d
        t_vel = kp_t*err_t
        send = self.make_twist(x_vel, t_vel)
        self.pub.publish(send)

    def run(self):
        """main run loop"""
        while self.go:
            # checks if list is populated by first scan
            if isinstance(self.xs, list):
                t_x, t_y = self.points_to_vector(self.xs, self.ys, 1, .2)
                r, theta = self.cart_to_polar(t_x, t_y)
                self.drive_to_target(r, theta)

        self.pub.publish(self.stop) # if bump sense breaks self.go loop, stop

if __name__ == '__main__':
    node = AvoidObject()
    node.run()
