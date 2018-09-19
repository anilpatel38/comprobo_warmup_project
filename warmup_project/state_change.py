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

class StateChange(object):

    def __init__(self):
        rospy.init_node("StateChange")
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.process_scan)
        self.stop = self.make_twist(0,0)
        self.xs = None
        self.ys = None
        self.xsf = None
        self.ysf = None
        self.state = "avoid" #states are "wall" and "avoid"

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
        xs = []
        ys = []
        xsf = []
        ysf = []
        for i in range(len(ranges)):
            theta = math.radians(i+90)
            r = ranges[i]
            xf = math.cos(theta)*r
            yf = math.sin(theta)*r
            xsf.append(xf)
            ysf.append(yf)
            if r != 0:
                xs.append(xf)
                ys.append(yf)

        self.xs = xs
        self.ys = ys
        self.xsf = xsf
        self.ysf = ysf


    def cart_to_polar(self, x, y):
        r = math.sqrt(y**2 + x**2)
        # subtract 90 to account for robot forward heading=90
        theta = math.degrees(numpy.arctan2(y,x))-90
        return (r, theta)

    def points_to_vector(self, x_list,y_list,a,radius):
        # a is scaling factor
        neg_vector = [0,0]
        s = 3 # spread of field
        for i in range(len(x_list)):
            x = x_list[i]
            y = y_list[i]
            dist = math.sqrt(y**2 + x**2)
            theta = math.atan2(y, x)
            x = a * (s + radius - dist) * math.cos(theta)
            y = a * (s + radius - dist) * math.sin(theta)
            neg_vector[0] = neg_vector[0] + x
            neg_vector[1] = neg_vector[1] + y
        pos_vector = [0, .5*a]

        return [pos_vector[0] + neg_vector[0], pos_vector[1] + neg_vector[1]]

    def ransac(x,y):
        """Finds most robust line.

        Args:
            x (list): list of integers that represent x values of points.
            y (lists): list of integers that represent y values of points.
        Returns:
            None if line is parallel or there is no line, slope if not.
        """
        n = len(x)
        threshold = .2
        xrange = abs(max(x)-min(x))
        yrange = abs(max(y)-min(y))
        threshold = threshold * statistics.mean([xrange, yrange])/20
        final_slope = None
        final_b = None
        maxcount = 0
        for i in range(n):
            random_indexes = random.choices(list(range(n)), k=2)
            i1 = random_indexes[0]
            i2 = random_indexes[1]

            x1 = x[i1]
            y1 = y[i1]

            x2 = x[i2]
            y2 = y[i2]
            run = x2 - x1
            if run == 0:
                slope = None
            else:
                slope = (y2 - y1) / run
                b = y1 - slope*x1

            count = 0;

            for m in range(1,n):
                p0 = (x[m-1],y[m-1])
                p = (x[m], y[m])
                dist_points = math.sqrt(math.pow((x[m]-x[m-1]),2)+math.pow((y[m]-y[m-1]),2))
                new_y = slope * x[m-1] + b
                dif_y = abs(new_y - y[m-1])

                if dist_points <= .15 and dif_y <= .05:
                    count += 1
                elif dist_points > .15 and count > 0:
                    break

            if count > maxcount:
                maxcount = count
                final_slope = slope
                final_b = b

        return final_slope, final_b, maxcount

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
                slope, intercept, maxcount = self.ransac(xs,ys)
                if maxcount > 20: # this is arbritary and should be tweaked
                    self.state = "wall"
                    # drive somehow
                else:
                    t_x, t_y = self.points_to_vector(xs, ys, 1, .2) #we can test these out
                    plt.plot(self.xs, self.ys, 'yo', markersize=10) # where robot should be going
                    plt.plot(t_x, t_y)
                    #plt.plot(self.xsf, self.ysf, 'yo', markersize=10)
                    r, theta = self.cart_to_polar(t_x, t_y)
                    print(r, theta)
                    self.drive_to_target(r, theta)
                # plt.show()

if __name__ == '__main__':
    node = AvoidObstacle()
    node.run()
