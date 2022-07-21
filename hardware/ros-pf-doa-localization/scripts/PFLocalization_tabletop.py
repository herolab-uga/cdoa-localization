import math
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as pb
import random
from datetime import datetime
import time
import csv
import sys




#!/usr/bin/env python
import os, sys, time
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

    log_on_file = False
    now = datetime.now()
    # filename = "combined_sync_data-{:d}".format(now.month) + '-' + "{:d}".format(now.day) + '_' + "{:d}".format(now.hour) + '-' + "{:d}".format(now.minute) + '-' + "{:d}".format(now.second) + ".csv"
    #filename = "/home/kilop/Desktop/OdometryWifi_datalog_" + "{:d}".format(now.month) + '-' + "{:d}".format(now.day) + '_' + "{:d}".format(now.hour) + '-' + "{:d}".format(now.minute)
 
    # file_log = open(filename, 'w')
    

    def __init__(self):
        self.sub_rssi = rospy.Subscriber('/node1/network_analysis/wireless_quality', WirelessLink, self.rssi1_callback, queue_size =10)
        self.sub_rssi = rospy.Subscriber('/node2/network_analysis/wireless_quality', WirelessLink, self.rssi2_callback, queue_size =10)
        self.sub_rssi = rospy.Subscriber('/node3/network_analysis/wireless_quality', WirelessLink, self.rssi3_callback, queue_size =10)
        self.sub_rssi = rospy.Subscriber('/node4/network_analysis/wireless_quality', WirelessLink, self.rssi4_callback, queue_size =10)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size =10)
        self.sub_vel = rospy.Subscriber('/cmd_vel', Twist, self.vel_callback, queue_size =10)
        self.sub_pos = rospy.Subscriber('/positions', Robot_Pos, self.pos_callback, queue_size =10)
 
    def rssi1_callback(self, data):
        #self.msg_vec = data
        self.temp_step = data.header.seq
        self.temp_stamp_sec = data.header.stamp.secs
        self.temp_stamp_nsec = data.header.stamp.nsecs
        self.rssi1 = data.rssi
        self.lqi1 = data.lqi
        self.noise1 = data.noise
        self.status1 = data.status
        self.log_on_file = True
        # print('true')
        if (self.log_on_file == True):
            string = "{:9d},".format(self.temp_step) +"{:11d},".format(self.temp_stamp_sec) +"{:11d},".format(self.temp_stamp_nsec)+ "{:10f},".format(self.robot_odom_x) + "{:10f},".format(self.robot_odom_y) +"{:10f},".format(self.robot_odom_w) + "{:10f},".format(self.robot_pos_x) + "{:10f},".format(self.robot_pos_y) +"{:10f},".format(self.robot_pos_w) + "{:10f},".format(self.vel_x) + "{:10f},".format(self.vel_y) + "{:10f},".format(self.vel_w) + "{:5f},".format(self.rssi1) + "{:5f},".format(self.lqi1) + "{:5f},".format(self.noise1) + "{:2d},".format(self.status1) + "{:5f},".format(self.rssi2) + "{:5f},".format(self.lqi2) + "{:5f},".format(self.noise2) + "{:2d},".format(self.status2) + "{:5f},".format(self.rssi3) + "{:5f},".format(self.lqi3) + "{:5f},".format(self.noise3) + "{:2d},".format(self.status3) + "{:5f},".format(self.rssi4) + "{:5f},".format(self.lqi4) + "{:5f},".format(self.noise4) + "{:2d}".format(self.status4) + "\n"
        # self.file_log.write (string)
    
        
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
        # self.file_log.write (string)


# pos= (4,7)
# pos = (1,14)
# Next Pos(-5,-1)
#freq GHz
#power in decibels per milliwatt (dBm)

def quaternion_to_yaw(x, y, w, z):
        t0 = 2.0 * (w * z + x * y)
        t1 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t0, t1)
        return yaw

DT = 0.1
def motion_model(x,u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2]), 0],
                  [DT * math.sin(x[2]), 0],
                  [0.0, DT]])

    xd = F.dot(x) + B.dot(u)

    return xd

def dist(x, y, pos):
    return math.sqrt((pos[0]-x)**2 + (pos[1]-y)**2)


rss0 = 20 * math.log10(3 / (4 * math.pi * 2.4 * 10))
rss0 = rss0-2*random.random()
print(rss0)
areaSize=(2.34, 1.75)

node_pos=[(0,0),(0,1.75),(2.34,1.75),(2.34,0)]

def gen_wifi(freq=2.4, power=20, trans_gain=0, recv_gain=0, size=areaSize, pos=(5,5), shadow_dev=2, n=3,noise=2):
    if pos is None:
        pos = (random.randrange(size[0]), random.randrange(size[1]))

    random.seed(datetime.now())
    
    normal_dist = np.random.normal(0, shadow_dev, size=[int(size[0])+1, int(size[1])+1])
    rss = []

    random.seed(datetime.now())

    for x in range(0,4):
        distance = dist(node_pos[x][0], node_pos[x][1], pos)
        val = rss0 - 10 * n * math.log10(distance) + normal_dist[int(pos[0])][int(pos[1])] if distance != 0 else rss0 + normal_dist[int(pos[0])][int(pos[1])]
        rss.append(val-noise*random.random())
        # print("rssi: "+str(val)+"\tnode_position: "+str(node_pos[x])+"\tPF_Pos: "+str(pos)+"\tdistance: "+str(distance)+"\t"+str(normal_dist[int(pos[0])][int(pos[1])]))
    return rss

doa=[]
def find_doa(overall_rss,original_tragectory,i):
    inner_curr = i
    limit = i-100 if i>100 else 0
    est_sin_sum = 0
    est_cos_sum = 0
    starting_curr = inner_curr
    weight_sum = 0
    # average estimated DoA calculated
    while inner_curr >= limit:
    # print(str(overall_rss[i]))
        gy = ((overall_rss[i][1]-overall_rss[i][0])/2) + ((overall_rss[i][2]-overall_rss[i][3])/2)
        gx = ((overall_rss[i][2]-overall_rss[i][1])/2) + ((overall_rss[i][3]-overall_rss[i][0])/2)
        estimated_grad=np.arctan(gy/gx) if gx!=0 else 0
        quat = quaternion_to_yaw(original_tragectory[i][0],original_tragectory[i][1], original_tragectory[i][2],original_tragectory[i][3])
        estimated_grad += quat
        if estimated_grad > math.pi:
            estimated_grad = -2 * math.pi + estimated_grad
        elif estimated_grad < -math.pi:
            estimated_grad = math.pi - abs(-math.pi - estimated_grad)
        weight = 0.99 ** (inner_curr - starting_curr)
        weight_sum += weight
        estimated_grad = weight * estimated_grad
        est_sin_sum += math.sin(estimated_grad)
        est_cos_sum += math.cos(estimated_grad)
        inner_curr -= 1
    avg_est_sin = est_sin_sum / weight_sum
    avg_est_cos = est_cos_sum / weight_sum
    avg_grad = math.atan2(avg_est_sin, avg_est_cos)
    doa.append(avg_grad)
    # if not prev:
    #     prev = (i,avg_grad)
    return avg_grad



def localize(rec):

    initial_pos=(1.199,0.0898)
    possible_x = list(np.arange(0.9,1.3,0.05))
    possible_y = list(np.arange(0.6,1,0.05))
    num_particles = 1000

    overall_rss=[]
    original_tragectory=[]
    velocity=[]
    Previous_pos = initial_pos


    plt.ion()

    random.seed(datetime.now())
    previous_errors =[]
    distance_error =[]
    particles = []
    times = []
    Previous_pos = initial_pos
    start_time = time.time()
    for x in range(num_particles):
        particles.append((random.choice(possible_x),random.choice(possible_y)))
    _ , plt_pos = plt.subplots(1,1)
    plt_pos.set_title("Localization")
    plt_pos.set_xlim((-0.25,areaSize[0]+0.25))
    plt_pos.set_ylim((-0.25,areaSize[1]+0.25))
    i = 0
    while(1):
        # print (rec.log_on_file)
        if rec.log_on_file == True:
            rec.log_on_file = False
            x, y, w, z = rec.robot_pos_x, rec.robot_pos_y, rec.robot_pos_w, 0
            original_tragectory.append((x,y,w,z))
            rss =  [rec.rssi1,rec.rssi2,rec.rssi3,rec.rssi4]
            overall_rss.append(rss)
            velocity.append([rec.vel_x,rec.vel_w])
            positions =[]
            errors=[]
            weights =[]
            actual_rss_ls=[]
            error=0
            for particle in particles:
                x,y=particle[0],particle[1]
                actual_rss = gen_wifi(pos=(x,y),noise=0)
                gy = ((actual_rss[1]-actual_rss[0])/2) + ((actual_rss[2]-actual_rss[3])/2)
                gx = ((actual_rss[2]-actual_rss[1])/2) + ((actual_rss[3]-actual_rss[0])/2)
                adoa=np.arctan(gy/gx) if gx !=0 else 0
                avg_doa=find_doa(overall_rss,original_tragectory,i)
                error=abs(adoa-avg_doa)
                # error=np.sum(np.subtract(actual_rss,overall_rss[i]))
            
                # std_error=1

                std_error=np.std(np.subtract(actual_rss,overall_rss[i]))
                omega=((1/((std_error)*math.sqrt(2*math.pi)))*(math.pow(math.e,-(math.pow(error,2)/(2*(std_error**2))))))
                for i in range(len(previous_errors)-1,len(previous_errors)-4 if len(previous_errors) > 5 else 0,-1):
                    omega=omega*((1/((std_error)*math.sqrt(2*math.pi)))*(math.pow(math.e,-(math.pow(previous_errors[i],2)/(2*(std_error**2))))))
                    
                weights.append(omega)
                positions.append((x,y,))
                errors.append(error)
                actual_rss_ls.append(actual_rss)
                
            sum_weight=np.sum(weights)
            if sum_weight == 0:
                pass
            for j in range(0,len(weights)):
                weights[j]=weights[j]/sum_weight


            max_weight = max(weights)
            max_index = weights.index(max_weight)
            pos = positions[max_index]
            pos = ((pos[0]+original_tragectory[i][0])/2,(pos[1]+original_tragectory[i][1])/2)
            previous_errors.append(errors[max_index])
            distance_error.append(dist(pos[0],pos[1],original_tragectory[i]))


            # print(str(pos)+"\t"+str(actual_rss_ls[max_index]))#+"\t"+str(errors[max_index])+"\t"+str(num_particles))
            # plt_pos.plot(pos[0],pos[1],'bo',markersize=3,clip_on=False)
            pos_x = [pos[0],pos[1],original_tragectory[i][2]]
            xd = motion_model(pos_x,velocity[i])
            # print(xd)

            num_particles=math.ceil(num_particles/2) if num_particles/2>200 else 200
            particles=[]
            for x in range(num_particles):
                particles.append((random.uniform(pos_x[0]-0.1 if pos_x[0]-0.1 >=-areaSize[0] else -areaSize[0], pos_x[0]+0.1 if pos_x[0]+0.1 <=areaSize[0] else areaSize[0])
                ,random.uniform(pos[1]-0.1 if pos_x[1]-0.1 >=-areaSize[1] else -areaSize[1], pos_x[1]+0.1 if pos_x[1]+0.1 <=areaSize[1] else areaSize[1])))
            if i>0:
                plt_pos.plot([original_tragectory[i-1][0],original_tragectory[i][0]],[original_tragectory[i-1][1],original_tragectory[i][1]],'g-',linewidth=2,clip_on=False)
                plt_pos.plot([Previous_pos[0],pos[0]],[Previous_pos[1],pos[1]],'r-',linewidth=2,clip_on=False)

                # plt_error.plot([i-1,i],[distance_error[i-1],distance_error[i]],'r-')
            plt.draw()
            plt.pause(0.0001)
            Previous_pos = pos
            i+=1

    

    plt.show(block=False)
    plt.savefig('predicted_trajectory_'+sys.argv[1]+'.png')
    print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
    rsscumulativeEror=np.sum(previous_errors)
    rssmeanError=np.average(previous_errors)
    rssStandardDeviationError=np.std(previous_errors)
    distcumulativeEror=np.sum(distance_error)
    distmeanError=np.average(distance_error)
    distStandardDeviationError=np.std(distance_error)
    print("RSS_ERROR:   Cumulative Error: " + str(rsscumulativeEror)+"\tMean  Error: "+str(rssmeanError)+"\tStandard Deviation: "+str(rssStandardDeviationError))
    print("DIST_ERROR:   Cumulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))
    # resultFile = open("error_field_"+str(NOISE_LEVEL)+"_"+str(RESOLUTION)+".csv", "a")  # append mode
    # resultFile.write(str(rsscumulativeEror)+","+str(rssmeanError)+","+str(rssStandardDeviationError)
    # +","+str(distcumulativeEror)+","+str(distmeanError)+","+str(distStandardDeviationError)+"\n")
    # resultFile.close()
    plt.pause(3)
    plt.clf()
    plt.close()


def main(args):
    '''Initializes and cleanup ros node'''
    rec = recorder_rssi_odom()
    rospy.init_node('record_rssi_odom_node', anonymous=True)
    # rec.function_init()
    localize(rec)
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print ("Shutting down ROS Image feature detector module")

if __name__ == '__main__':
    main(sys.argv)

