#!/usr/bin/env python
import os, sys, time, datetime
import rospy 
from network_analysis.msg import *
import std_msgs.msg
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from robot_msgs.msg import Robot_Pos

class recorder_rssi_odom:

    msg_vec = WirelessLinkVector()
    odom_val = Odometry()
    vel = Twist()


    temp_step = 0
    temp_stamp_sec = 0
    temp_stamp_nsec = 0
    robot_pos_x = 0
    robot_pos_y = 0
    robot_odom_x = 0
    robot_odom_y = 0
    robot_odom_w = 0
    robot_q_x = 0
    robot_q_y = 0
    robot_q_z = 0
    robot_pos_w = 0
    robot_vel_x = 0
    robot_vel_y = 0
    robot_vel_w = 0
    vel_x = 0
    vel_y = 0
    vel_w = 0
    rssi1 = 0
    lqi1 = 0
    noise1 = 0
    status1 = False
    rssi2 = 0
    lqi2 = 0
    noise2 = 0
    status2 = False
    rssi3 = 0
    lqi3 = 0
    noise3 = 0
    status3 = False
    rssi4 = 0
    lqi4 = 0
    noise4 = 0
    status4 = False

    log_on_file = True
    now = datetime.datetime.now()
    filename = "combined_sync_data-{:d}".format(now.month) + '-' + "{:d}".format(now.day) + '_' + "{:d}".format(now.hour) + '-' + "{:d}".format(now.minute) + '-' + "{:d}".format(now.second) + ".csv"
    #filename = "/home/kilop/Desktop/OdometryWifi_datalog_" + "{:d}".format(now.month) + '-' + "{:d}".format(now.day) + '_' + "{:d}".format(now.hour) + '-' + "{:d}".format(now.minute)
 
    file_log = open(filename, 'w')
    

    def __init__(self):
        self.sub_rssi = rospy.Subscriber('/node1/network_analysis/wireless_quality', WirelessLink, self.rssi1_callback, queue_size =1)
        self.sub_rssi = rospy.Subscriber('/node2/network_analysis/wireless_quality', WirelessLink, self.rssi2_callback, queue_size =1)
        self.sub_rssi = rospy.Subscriber('/node3/network_analysis/wireless_quality', WirelessLink, self.rssi3_callback, queue_size =1)
        self.sub_rssi = rospy.Subscriber('/node4/network_analysis/wireless_quality', WirelessLink, self.rssi4_callback, queue_size =1)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size =1)
        self.sub_vel = rospy.Subscriber('/cmd_vel', Twist, self.vel_callback, queue_size =1)
        self.sub_pos = rospy.Subscriber('/positions', Robot_Pos, self.pos_callback, queue_size =1)
 
    def rssi1_callback(self, data):
        #self.msg_vec = data
        self.temp_step = data.header.seq
        self.temp_stamp_sec = data.header.stamp.secs
        self.temp_stamp_nsec = data.header.stamp.nsecs
        self.rssi1 = data.rssi
        self.lqi1 = data.lqi
        self.noise1 = data.noise
        self.status1 = data.status

        if (self.log_on_file == True):
            string = "{:9d},".format(self.temp_step) +"{:11d},".format(self.temp_stamp_sec) +"{:11d},".format(self.temp_stamp_nsec)+ "{:10f},".format(self.robot_odom_x) + "{:10f},".format(self.robot_odom_y) +"{:10f},".format(self.robot_odom_w) + "{:10f},".format(self.robot_pos_x) + "{:10f},".format(self.robot_pos_y) +"{:10f},".format(self.robot_pos_w) + "{:10f},".format(self.vel_x) + "{:10f},".format(self.vel_y) + "{:10f},".format(self.vel_w) + "{:5f},".format(self.rssi1) + "{:5f},".format(self.lqi1) + "{:5f},".format(self.noise1) + "{:2d},".format(self.status1) + "{:5f},".format(self.rssi2) + "{:5f},".format(self.lqi2) + "{:5f},".format(self.noise2) + "{:2d},".format(self.status2) + "{:5f},".format(self.rssi3) + "{:5f},".format(self.lqi3) + "{:5f},".format(self.noise3) + "{:2d},".format(self.status3) + "{:5f},".format(self.rssi4) + "{:5f},".format(self.lqi4) + "{:5f},".format(self.noise4) + "{:2d}".format(self.status4) + "\n"
        self.file_log.write (string)
        
    def rssi2_callback(self, data):
        #self.msg_vec = data
        self.rssi2 = data.rssi
        self.lqi2 = data.lqi
        self.noise2 = data.noise
        self.status2 = data.status

    def rssi3_callback(self, data):
        #self.msg_vec = data
        self.rssi3 = data.rssi
        self.lqi3 = data.lqi
        self.noise3 = data.noise
        self.status3 = data.status

    def rssi4_callback(self, data):
        #self.msg_vec = data
        self.rssi4 = data.rssi
        self.lqi4 = data.lqi
        self.noise4 = data.noise
        self.status4 = data.status

    def odom_callback(self,data):
        #self.odom_val = data
        self.robot_odom_x = data.pose.pose.position.x
        self.robot_odom_y = data.pose.pose.position.y
        # self.robot_q_x = data.pose.pose.orientation.x
        # self.robot_q_y = data.pose.pose.orientation.y
        # self.robot_q_z = data.pose.pose.orientation.z
        self.robot_odom_w = data.pose.pose.orientation.w
        # self.robot_vel_x = data.twist.twist.linear.x 
        # self.robot_vel_y = data.twist.twist.linear.y
        # self.robot_vel_w = data.twist.twist.angular.z

    def pos_callback(self,data):
        if data.robot_pos:
            self.robot_pos_x = data.robot_pos[0].pose.pose.position.x
            self.robot_pos_y = data.robot_pos[0].pose.pose.position.y
            # self.robot_q_x = data.pose.pose.orientation.x
            # self.robot_q_y = data.pose.pose.orientation.y
            # self.robot_q_z = data.pose.pose.orientation.z
            self.robot_pos_w = data.robot_pos[0].pose.pose.orientation.w
            # self.robot_vel_x = data.twist.twist.linear.x 
            # self.robot_vel_y = data.twist.twist.linear.y
            # self.robot_vel_w = data.twist.twist.angular.z

    def vel_callback(self,data):
        self.vel_x = data.linear.x
        self.vel_y = data.linear.y
        self.vel_w = data.angular.z

    def function_init(self):
        if (self.log_on_file == True):
            string ="temp_step,temp_sec,temp_nsec,robot_odom_x,robot_odom_y,robot_odom_w,robot_pos_x,robot_pos_y,robot_pos_w,vel_x,vel_y,vel_w,rssi1,lqi1,noise1,status1,rssi2,lqi2,noise2,status2,rssi3,lqi3,noise3,status3,rssi4,lqi4,noise4,status4\n"
        self.file_log.write (string)



def main(args):
    '''Initializes and cleanup ros node'''
    rec = recorder_rssi_odom()
    rospy.init_node('record_rssi_odom_node', anonymous=True)
    rec.function_init()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")

if __name__ == '__main__':
    main(sys.argv)
