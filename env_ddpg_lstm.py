#!/usr/bin/env python2
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Cameron #
import rospy
import roslaunch
import time
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import Image   #Dont forget 'Image'
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn
#import matplotlib.pyplot as plt
# import skimage as skimage
# from skimage import transform, color, exposure
# from skimage.transform import rotate
# from skimage.viewer import ImageViewer

maxloop = 15

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.scan = []
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.goal_distance = 0
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.getScan) #subscribe the scan data
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 3
        self.frame_skip = 4
        self.jump = 0
        self.last50actions = [0] * 50
 
    def getScan(self,scan):
        scan_range = []
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])
        self.scan = scan_range
        
        
    def getGoalDistance(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.goal_distance = goal_distance
        return goal_distance

    def getOdometry(self, odom):#get the position
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)


    def judgeCollision_Reach(self) :
        collision = False
        goal = False
        min_range = 0.21
        if min_range > min(self.scan) > 0:
            collision = True
            
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        self.goal_distance = current_distance
        if current_distance < 0.2:
            self.get_goalbox = True
            goal = True
        return collision, goal
    
    def setReward(self, collision,action):
        
        #reward1: orient difference
        #heading = -math.fabs(self.heading)
        heading = self.heading
        #print " heading:",  heading
        r_yaw = heading
        reward = r_yaw
        standard_action = False
        #reward2: colllision
        if collision:
            rospy.loginfo("Collision!!")
            reward = -200
            self.pub_cmd_vel.publish(Twist())
            self.jump += 1
        #reward3: goal
        elif self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 150
            self.jump = 0
            self.pub_cmd_vel.publish(Twist())
            self.initGoal = True
            # self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            # self.goal_distance = self.getGoalDistance()
            self.get_goalbox = False
      
        else :
            #reward4: goal_distance
            reward += -5 * self.goal_distance
            #reward5: loop avoidance:
            action_sum = sum(self.last50actions)
            if action_sum > 45:
                reward += -50
            #reward6: forward,left,right
            if round(action[0],1) == 0.2 and round(action[1]) == 0:
                reward += 20
                standard_action = True
            if round(action[0],2) == 0.05 and round(action[1],1) == 0.2:
                reward += 10
                standard_action = True
            if round(action[0],2) == 0.05 and round(action[1],1) == -0.2:
                reward += 10
                standard_action = True
            
        rospy.loginfo("reward: %f, goal_dist : %f", reward, self.goal_distance)
        if standard_action:
            print "standard_action"
    
        if self.jump >= maxloop:
            self.initGoal = True
            self.jump = 0
        return reward

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # Send action command
    
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        #rospy.loginfo("linear: %f, angular:%f",action[0], action[1])
        for i in range(self.frame_skip):
            self.pub_cmd_vel .publish(vel_cmd)
        
        # loop avoidance
        self.last50actions.pop(0) #remove oldest
        if action[1] < 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)
  

        success=False
        # k frames skipping : skip the k - 1 frames and only use the kth frame
        for i in range(self.frame_skip):
            image_data = None
            while image_data is None or success is False:
                try:
                    image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                    h = image_data.height
                    w = image_data.width
                    cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                    if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                        success = True
                    else:
                        pass
                except:
                    pass
            #Preprocess the image
            '''x_t = skimage.color.rgb2gray(cv_image)
            x_t = skimage.transform.resize(x_t,(32,32))
            x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''
            if i == self.frame_skip - 1:
               # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                cv_image = cv2.GaussianBlur(cv_image,(5,5),0)
                cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
                cv_image= cv_image.reshape(3, cv_image.shape[0], cv_image.shape[1]) #(1 batchsize, )3 channel, rows, cols
        state_img =np.array(cv_image)
        
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        # # Add center of the track reward
        # # len(data.ranges) = 100
        # laser_len = len(data.ranges)
        # left_sum = sum(data.ranges[laser_len-(laser_len/5):laser_len-(laser_len/10)]) #80-90
        # right_sum = sum(data.ranges[(laser_len/10):(laser_len/5)]) #10-20

        # center_detour = abs(right_sum - left_sum)/5

        #judge collision or reach
        collision, goal = self.judgeCollision_Reach()
        
        #Calculate the reward 
        reward  = self.setReward(collision, action)
       
        done = 0
        if collision: 
            done = 1
        if goal:
            done = 2
        #state_pos = np.array([self.position.x, self.position.y,  self.goal_x, self. goal_y])
        cur_dis = self.getGoalDistance()
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        state_pos = np.array([cur_dis, goal_angle])
        return state_img, state_pos, reward, done,  {} 


    def reset(self):
        self.last50actions = [0] * 50 #used for looping avoidance

        # Resets the state of the environment and returns an initial observation.
        rospy.loginfo("reset ")
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            rospy.loginfo ("/gazebo/reset_simulation service call failed")
            
       # rospy.loginfo("reset : 2")
        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            rospy.loginfo ("/gazebo/unpause_physics service call failed")
          
        #rospy.loginfo("reset : 3")  
        #Initial the goal position and distance between the goal and robot
        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(position_check=True, delete=True)
            self.initGoal = False
        self.goal_distance = self.getGoalDistance()
        
        #rospy.loginfo("reset : 4")
        #Get the initial camera state
        success=False
        #image_date = None
        for i in range(self.frame_skip): # k frames skipping : skip the k - 1 frames and only use the kth frame
            image_data = None
            while image_data is None or success is False:
                try:
                    image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                    h = image_data.height
                    w = image_data.width
                    cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                    #rospy.loginfo("Capture the image!")
                    #temporal fix, check image is not corrupted
                    if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                        success = True
                    else:
                        pass
                        #print("/camera/rgb/image_raw ERROR, retrying")
                except:
                    pass
            #rospy.loginfo("finish capture the image")
            #Preprocess the image
            '''x_t = skimage.color.rgb2gray(cv_image)
            x_t = skimage.transform.resize(x_t,(32,32))
            x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''
            if i == self.frame_skip - 1:
                #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                cv_image = cv2.GaussianBlur(cv_image,(5,5),0)
                cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
                #cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
                #cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))
                cv_image= cv_image.reshape(3, cv_image.shape[0], cv_image.shape[1]) #(1 batchsize, )1 channel, rows, cols
        state_img =np.array(cv_image)
        #state_pos = np.array([self.position.x, self.position.y,  self.goal_x, self. goal_y])
        cur_dis = self.getGoalDistance()
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        state_pos = np.array([cur_dis, goal_angle])
        return state_img, state_pos

