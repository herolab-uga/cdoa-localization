import math
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as pb
import random
from datetime import datetime
import time
import sys
import csv

def dist(x, y, pos):
    return math.sqrt((pos[0]-x)**2 + (pos[1]-y)**2)


areaSize=(4, 4)
node_pos = [(0,0),(0,4),(4,0)]

if 'wifi' in sys.argv[1]:
    rss0 = -45.73
    pathloss_exponent = 2.162
elif 'ble' in sys.argv[1]:
    rss0 = -75.48
    pathloss_exponent = 2.271 
elif 'zigbee' in sys.argv[1]:
    rss0 = -50.33
    pathloss_exponent = 2.935


with open(sys.argv[1]) as f:
    dict_from_csv = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]

overall_rss=[]
original_tragectory=[]

for i in range(len(dict_from_csv)):
    dict=dict_from_csv[i]
    x , y = float(dict['x']) , float(dict['y'])
    original_tragectory.append((x,y))
    random.seed(datetime.now())

    rss = [-int(float(dict['RSSI A']))-random.random(),-int(float(dict['RSSI B']))-random.random() ,-int(float(dict['RSSI C']))-random.random()]
    overall_rss.append(rss)



def calc_dist(rss):
    cal_d= pow(10,((rss-rss0)/(-10*pathloss_exponent)))
    return cal_d

def wcl(weight,x,y):
    # print(x,y)
    xiwi=np.multiply(x,weight)
    yiwi=np.multiply(y,weight)
    xw=np.sum(xiwi)/np.sum(weight)
    yw=np.sum(yiwi)/np.sum(weight)
    return xw,yw

distance_error =[]
start_time = time.time()

for i in range(0,len(original_tragectory)):
    weight_arr = []
    x = overall_rss[i]
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    cal_d=calc_dist(x1)
    weight_arr=np.append(weight_arr,(1/cal_d))
    cal_d=calc_dist(x2)
    weight_arr=np.append(weight_arr,(1/cal_d))
    cal_d=calc_dist(x3)
    weight_arr=np.append(weight_arr,(1/cal_d))
    res_x,res_y=wcl(weight_arr,[node_pos[j][0] for j in range(0,3)],[node_pos[j][1] for j in range(0,3)]) 
    distance_error.append(dist(res_x,res_y,original_tragectory[i]))


print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
distcumulativeEror=np.sum(distance_error)
distmeanError=np.average(distance_error)
distStandardDeviationError=np.std(distance_error)
print("DIST_ERROR:   Cumulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))
resultFile = open("error_boundry_WC.csv", "a")  # append mode


