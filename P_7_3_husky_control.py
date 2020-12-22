# Author: Sisi(Scarlett) Liu <scarlett.liu@csu.edu.cn>
# Uni: Central South University, Changsha, China
# Online course: Robotic Vision and Applications
# Created: 09-Sep-2020   Modified: 09-Sep-2020   Revision: 1.0
# Copyright � 2020 SR2V Ltd. All rights reserved

# Lecture 7-3 Husky Control

import rospy, tf
import geometry_msgs.msg, nav_msgs.msg
from math import *

def huskyOdomCallback(message,cargs):
    # Implementation of proportional position control
    # For comparison to Simulink implementation
    # Callback arguments
    pub,msg,goal = cargs
    # Tunable parameters
    wgain = 5.0 # Gain for the angular velocity [rad/s / rad]
    vconst = 1.0 # Linear velocity when far away [m/s]
    distThresh = 0.5 # Distance treshold [m]
    # Generate a simplified pose
    pos = message.pose.pose
    quat = pos.orientation
    # From quaternion to Euler
    angles = tf.transformations.euler_from_quaternion((quat.x,quat.y,
                                                       quat.z,quat.w))
    theta = angles[2]
    pose = [pos.position.x, pos.position.y, theta]  # X, Y, Theta

    # Proportional Controller
    v = 0 # default linear velocity
    w = 0 # default angluar velocity
    distance = sqrt((pose[0]-goal[0])**2+(pose[1]-goal[1])**2)
    if (distance > distThresh):
        v = vconst
        desireYaw = atan2(goal[1]-pose[1],goal[0]-pose[0])
        u = desireYaw-theta
        bound = atan2(sin(u),cos(u))
        w = min(0.5 , max(-0.5, wgain*bound))

    # Publish
    msg.linear.x = v
    msg.angular.z = w
    pub.publish(msg)

    # Reporting
    print('huskyOdomCallback: x=%4.1f,y=%4.1f dist=%4.2f, cmd.v=%4.2f, cmd.w=%4.2f\n'%(pose[0],pose[1],distance,v,w))
########################################
# Main Script
# Initialize our node
# rospy.init_node('nre_simhuskycontrol',anonymous=True)
rosinit('127.0.0.1') #

# getting photo from your local camera

# image processing

# transfer x/y in pixel coords to world coords


# Set waypoint for Husky to drive to
goal = [x,y]  # Goal position in x/y ---> changable
# Setup publisher
cmdmsg = geometry_msgs.msg.Twist()
cmdpub = rospy.Publisher('/cmd_vel',geometry_msgs.msg.Twist, queue_size=10)
# Setup subscription - which implemets our controller.
# We pass the publisher, the message to publish and the goal as
# additional parameters to the callback function.
rospy.Subscriber('odom',nav_msgs.msg.Odometry,huskyOdomCallback,
                 (cmdpub,cmdmsg,goal))
# spin() simply keeps python from exiting until this node is stopped
rospy.spin()
