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


areaSize=(10, 10)
node_pos=[(0,0),(10,0),(10,10),(0,10)]

rssi_dict = []
for i in range(4):
    with open(sys.argv[1]+"s"+str(i)+".csv") as f:
        dict_from_csv = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f,delimiter=';', skipinitialspace=True)]
    rssi_dict.append(dict_from_csv)

min_length = len(rssi_dict[0])
for i in range(1,4):
    if len(rssi_dict[i]) < min_length:
        min_length = len(rssi_dict[i])

RSS0 = -47
overall_rss=[]
original_tragectory=[]
path_loss_list = []
received_signal_log = []
for i in range(min_length):
    x , y = float(rssi_dict[0][i]['x']) , float(rssi_dict[0][i]['y'])
    # if (x,y) != Previous_pos:
    original_tragectory.append((x,y))
    random.seed(datetime.now())
  
    rss = [int(rssi_dict[0][i]['rssi'])-random.random(),int(rssi_dict[1][i]['rssi'])-random.random(),int(rssi_dict[2][i]['rssi'])-random.random(),int(rssi_dict[3][i]['rssi'])-random.random()]
    overall_rss.append(rss)
    if float(rssi_dict[0][i]['distance']) > 1.4 and float(rssi_dict[0][i]['distance']) < 1.5 :
        RSS0 = int(rssi_dict[0][i]['rssi'])
    elif float(rssi_dict[1][i]['distance']) > 1.4 and float(rssi_dict[1][i]['distance']) < 1.5:
        RSS0 = int(rssi_dict[1][i]['rssi'])
    elif float(rssi_dict[2][i]['distance']) > 1.4 and float(rssi_dict[2][i]['distance']) < 1.5 :
        RSS0 = int(rssi_dict[2][i]['rssi'])
    elif float(rssi_dict[3][i]['distance']) > 1.4 and float(rssi_dict[3][i]['distance']) < 1.5 :
        RSS0 = int(rssi_dict[3][i]['rssi'])
    for j in range(4):
        path_loss_list.append(20-rss[j])
        received_signal_log.append(10*math.log10(float(rssi_dict[j][i]['distance'])))
    # Previous_pos = (x,y)

average_path_loss =  np.average(path_loss_list)
average_received_signal_log =  np.average(received_signal_log)
nominator = 0
demonimator = 0
for i in range(len(path_loss_list)):
    nominator += (path_loss_list[i] - average_path_loss)*(received_signal_log[i] - average_received_signal_log)
    demonimator += math.pow((received_signal_log[i] - average_received_signal_log),2)
pathloss_exponent = nominator / demonimator


def getDistanceFromRSS(rssi):
    return math.pow(10,((RSS0-rssi)/(10*pathloss_exponent)))

dist_i = []
candidate_pos = []
for i in range(0,len(overall_rss)):
    dist_j = []
    for j in range(0,4):
        dist_j.append(getDistanceFromRSS(overall_rss[i][j]))
    dist_i.append(dist_j)
    candidate_pos_j =[]
    for j in range(0,4):
        y = node_pos[j][1]-dist_j[j]
        x = node_pos[j][0]
        candidate_inter_pos = []
        while y < node_pos[j][1]:
            x_inter = math.sqrt(abs((dist_j[j]**2) - ((y-node_pos[j][1])**2)))
            candidate_inter_pos.append((x_inter+x,y))
            candidate_inter_pos.append((x_inter+x,-y))
            candidate_inter_pos.append((-x_inter+x,y))
            candidate_inter_pos.append((-x_inter+x,-y))
            y+=1
        candidate_pos_j.append(candidate_inter_pos)
    candidate_pos.append(candidate_pos_j)

distance_error =[]
times  = []
for i in range(0,len(original_tragectory)):
    start_time = time.time()
    positions =[]
    errors=[]
    for j in range(0,4):
        for k in range(len(candidate_pos[i][j])):
            position = candidate_pos[i][j][k]
            error = 0
            for l in range(0,4):
                error_inter = math.sqrt(((position[0]-node_pos[l][0])**2) + ((position[1]-node_pos[l][1])**2))
                error = error + math.pow((error_inter - dist_i[i][l]),2)
            errors.append(error)
            positions.append(position)
    min_error = min(errors)
    min_index = errors.index(min_error)
    predicted_pos = positions[min_index]

    distance_error.append(dist(predicted_pos[0],predicted_pos[1],original_tragectory[i]))
    times.append(time.time() - start_time)
       
distcumulativeEror=np.sum(distance_error)
distmeanError=np.average(distance_error)
distStandardDeviationError=np.std(distance_error)
print("--- Average Computation Time per Iteration : %s seconds ---" % (np.average(times)))
print("rss0",RSS0,"path loss exponent: ",pathloss_exponent)
# print("RSS_ERROR:   Cumulative Error: " + str(rsscumulativeEror)+"\tMean  Error: "+str(rssmeanError)+"\tStandard Deviation: "+str(rssStandardDeviationError))
print("DIST_ERROR:   Cummulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))
