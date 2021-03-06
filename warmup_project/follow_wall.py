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

deg2rad = math.pi/180

class WallFollowing(object):
    """node for identifying a wall with RANSAC and orientating the robot to
       drive parallel to it. Assume the robot is placed near the wall at some
       offset angle."""

    def __init__(self):
        rospy.init_node("LaserTesting")
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.process_scan)
        self.stop = self.make_twist(0,0)
        self.xs = None
        self.ys = None
        self.slope = None

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

    def turn(self, a):
        """a is angle in degrees """
        print(a)
        angle_vel = 28.23 # degrees per second
        turn_time = math.fabs(a/28.23)
        dir = numpy.sign(a)
        twist = self.make_twist(0, .5*dir)
        start = datetime.now()
        self.pub.publish(twist)
        time.sleep(turn_time)

        self.pub.publish(self.stop)     # stop after desired turning

    def move_dist(self, distance):
        """takes a distance in meters and moves it forward. Works under the
        timing that 0.5 cmd_vel = 1 ft/s."""
        speed = 0.5
        m2ft = 0.3048
        dist_ft = distance/m2ft
        sec = dist_ft
        start = datetime.now()
        go = self.make_twist(speed, 0)
        self.pub.publish(go)
        time.sleep(sec)

        self.pub.publish(self.stop)     # stop after desired movement

    def process_scan(self, m):
        """callback function for the laser scan data. translates to xs and ys
           in a list for later functionality."""
        ranges = m.ranges
        xs = []
        ys = []

        for i in range(len(ranges)):
            if ranges[i] != 0:          # clean out 0 values
                theta = deg2rad*(i+90)  # at 90 to get 0 on x axis
                r = ranges[i]
                x = math.cos(theta)*r   # polar to cartesian
                y = math.sin(theta)*r
                xs.append(x)
                ys.append(y)

        self.xs = xs
        self.ys = ys

    def ransac(self, x,y):
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
        maxcount = 0
        for i in range(20):
            random_indexes = numpy.random.randint(0,n,size=2)
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
                print(final_slope)

        return final_slope

    def show_plot(self):
        """plotting laser scan data and the neato for debugging."""
        xs_r = self.xs
        ys_r = self.ys

        plt.plot(xs_r, ys_r, 'ro')
        plt.plot(0,0, 'bo', markersize=15)
        plt.show()

    def turn_theta(self):
        """takes the slop and finds the angle difference between the wall and
           the neato's heading. Turns appropriately to be parallel w/ wall."""
        xs_r = self.xs
        ys_r = self.ys

        # check if first laser scan has instantiated the self.xs list.
        if isinstance(xs_r, list) and len(xs_r)!=0:
            self.slope = self.ransac(xs_r, ys_r)

        if self.slope != None:
            if math.fabs(self.slope)>0:
                theta_r = math.atan(1/self.slope)
                theta_d = -theta_r/deg2rad

                self.turn(theta_d)
                self.move_dist(0.25)

    def run(self):
        while True:
            self.turn_theta()

if __name__ == '__main__':
    node = WallFollowing()
    node.run()
