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

class MakeSquare(object):
    """This node is to test the neato interface and get it moving"""

    def __init__(self):
        rospy.init_node("Drive")
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/bump', Bump, self.process_bump)
        self.stop = self.make_twist(0,0)
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
        send = Twist()
        send.linear.x = x
        send.linear.y = 0
        send.linear.z = 0
        send.angular.x = 0
        send.angular.y = 0
        send.angular.z = theta
        return send

    def turn(self, a):
        """ a is angle in degrees """

        a_vel = 28.23 # degrees per second
        sec = a/a_vel

        twist = self.make_twist(0, .5)
        self.pub.publish(twist)

        while(1):
            delta_t = datetime.datetime.now()-start
            delta_s = delta_t.total_seconds()
            if delta_s > sec:
                break

        self.pub.publish(stop)

    def move_dist(self, distance):
        """ Takes a distance in meters and moves it forward. Works under the
        timing that 0.5 cmd_vel = 1 ft/s."""

        speed = 0.5
        m2ft = 0.3048
        dist_ft = distance/m2ft
        sec = dist_ft

        start = datetime.datetime.now()
        go = self.make_twist(speed, 0)
        self.pub.publish(go)

        while(1):
            delta_t = datetime.datetime.now()-start
            delta_s = delta_t.total_seconds()
            if delta_s > sec:
                break

        self.pub.publish(self.stop)

    def drive_square(self, len):
        for i in range()

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
                move = [0,0]

                if key == 'w':
                    move = [1,0]
                if key == 's':
                    move = [-1,0]
                if key == 'p':
                    move = [0,0]
                if key == 'd':
                    move = [0,1]
                if key == 'a':
                    move = [0,-1]
                if key == 'e':
                    self.move_dist(0.5)
                if key == 'r':
                    self.turn

                print(move)

                speed = 0.5
                x_vel = move[0]*speed
                t_vel = move[1]*speed

                send = Twist()
                send.linear.x = x_vel
                send.linear.y = 0
                send.linear.z = 0
                send.angular.x = 0
                send.angular.y = 0
                send.angular.z = t_vel

                self.pub.publish(send)


if __name__ == '__main__':
    settings = termios.tcgetattr(sys.stdin)
    node = MakeSquare()
    node.run()
