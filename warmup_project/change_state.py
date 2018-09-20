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

class ChangeState(object):
    """state switching between wall following and obstacle avoidance. The state
       starts with avoiding obstacles until RANSAC finds a group of points that
       suitably fit a wall, which switches the state to wall following."""

    def __init__(self):
        # initialize ROS things - subs/pubs/etc.
        rospy.init_node("AvoidObject")
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.process_scan)
        rospy.Subscriber('/bump', Bump, self.process_bump)

        self.stop = self.make_twist(0,0)    # pre-made stop message
        self.xs = None                      # list of all xs from lidar
        self.ys = None                      # all ys from lidar
        self.go = True                      # used to watch the bump sensor

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
        """plot all lidar points and the neato in the neato ref frame"""
        plt.plot(self.xs, self.ys, 'ro')
        plt.plot(0,0, 'bo', markersize=15)
        #plt.show()

    def process_scan(self, m):
        """callback function triggered on the laser scan subscriber. cleans out
           all 0 values and only logs points within a range."""
        max_r = 1.5
        ranges = m.ranges
        xs = []
        ys = []
        xsf = []
        ysf = []
        for i in range(len(ranges)):
            if ranges[i] != 0 and ranges[i]<max_r:
                theta = math.radians(i+90)
                r = ranges[i]
                xf = math.cos(theta)*r
                yf = math.sin(theta)*r
                xs.append(xf)
                ys.append(yf)

        self.xs = xs
        self.ys = ys

    def process_bump(self, m):
        """callback function triggered on the bump subscriber. Changes self.go
           to break out of the run loop and stop the robot/program when the
           bump sensor is triggered. Convenient stopping mechanism."""
        lf = m.leftFront
        rf = m.rightSide

        # if either bunp sensor is triggered
        if lf != 0 or rf != 0:
            self.pub.publish(self.stop)
            self.go = False

    def cart_to_polar(self, x, y):
        """used to turn a goal point or vector into r and theta values for
           controlling the neato."""
        r = math.sqrt(y**2 + x**2)

        # subtract 90 to account for robot forward heading=90
        theta = math.degrees(numpy.arctan2(y,x))-90
        return (r, theta)

    def points_to_vector(self, x_list,y_list,a,radius):
        """Takes in a point cloud of obstacles and defines a vector at (0, 0)
           on the neato's position that points along the gradient of the
           potential field."""
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
        pos_vector = [0, 0.25*a]
        vector_move = [pos_vector[0] + neg_vector[0], pos_vector[1] + neg_vector[1]];

        return vector_move

    def ransac(self, x, y):
        """Finds most robust line.

        Args:
            x (list): list of integers that represent x values of points.
            y (lists): list of integers that represent y values of points.
        Returns:
            None if line is parallel or there is no line, slope if not.
        """
        n = len(x)
        print(n)
        threshold = .2
        xrange = abs(max(x)-min(x))
        yrange = abs(max(y)-min(y))
        threshold = threshold * statistics.mean([xrange, yrange])/20
        final_slope = None
        final_b = None
        maxcount = 0
        for i in range(n):
            random_indexes = numpy.random.randint(0,n,size=2)
            i1 = random_indexes[0]
            i2 = random_indexes[1]

            x1 = x[i1]
            y1 = y[i1]

            x2 = x[i2]
            y2 = y[i2]
            run = x2 - x1
            if run == 0:            # if undefined slope
                slope = 0.001       # set to super low number to not kill math
                b = y1 - slope*x1
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
                maxcount = count     # check number of points in the line
                final_slope = slope
                final_b = b

        return final_slope, final_b, maxcount

    def turn(self, a):
        """turns a specified angle, a is angle in degrees."""
        angle_vel = 28.23 # degrees per second, experimentally timed
        turn_time = math.fabs(a/28.23)
        dir = numpy.sign(a)
        twist = self.make_twist(0, .5*dir)
        start = datetime.now()
        self.pub.publish(twist)
        time.sleep(turn_time)

        self.pub.publish(self.stop)

    def move_dist(self, distance):
        """takes a distance in meters and moves it forward. Works under the
            experimental timing that 0.5 cmd_vel = 1 ft/s."""
        speed = 0.5
        m2ft = 0.3048
        dist_ft = distance/m2ft
        sec = dist_ft
        start = datetime.now()
        go = self.make_twist(speed, 0)
        self.pub.publish(go)            # send go for specified time
        time.sleep(sec)

        self.pub.publish(self.stop)

    def turn_theta(self, slope):
        """takes a wall slope from RANSAC, calculates angle difference between
           the the wall and the robots heading. Turns and drives robot."""

        if slope != None:
            if math.fabs(slope)>0:
                theta_r = math.atan(1/slope)
                theta_d = -math.degrees(theta_r)
                self.turn(theta_d)      # required angle to turn (degrees)
                self.move_dist(0.25)    # move forward some arbitrary amount.

    def drive_to_target(self, r, t):
        """proportional control for driving a robot towards a goal point in
           polar coordinates. Experiemtnally tuned Kp values."""
        goal_d = 0      # desired distance away from target
        goal_t = 0      # desired angle away from goal, 0 to face target

        err_d = r - goal_d  # error terms
        err_t = t - goal_t

        kp_d = 0.005      # proportional control constants
        kp_t = 0.01

        x_vel = kp_d*err_d  # velocities to send
        t_vel = kp_t*err_t

        send = self.make_twist(x_vel, t_vel)
        self.pub.publish(send)

    def run(self):
        """main run loop for the node. Checks that the first laser scan has
           instantiated the first list of points and also that the scan didn't
           yield an empty list, meaning the robot is in empty space."""
        while self.go:
            if isinstance(self.xs, list) and len(self.xs)!=0:
                slope, intercept, maxcount = self.ransac(self.xs, self.ys)

                if maxcount > 40: # if sufficient match to best RANSAC line
                    self.turn_theta(slope)
                else:   # else run obstacle avoidance
                    t_x, t_y = self.points_to_vector(self.xs, self.ys, 1, .2)
                    r, theta = self.cart_to_polar(t_x, t_y)
                    self.drive_to_target(r, theta)

        self.pub.publish(self.stop) # break run loop, if bump sensor is hit

if __name__ == '__main__':
    node = ChangeState()
    node.run()
