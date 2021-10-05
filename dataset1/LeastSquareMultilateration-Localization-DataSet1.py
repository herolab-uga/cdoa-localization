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

def getDistanceFromRSS(rssi):
    return math.pow(10,((rss0-rssi)/(10*pathloss_exponent)))

dist_i = []
candidate_pos = []
for i in range(0,len(overall_rss)):
    dist_j = []
    for j in range(0,3):
        dist_j.append(getDistanceFromRSS(overall_rss[i][j]))
    dist_i.append(dist_j)
    candidate_pos_j =[]
    for j in range(0,3):
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
start_time = time.time()


for i in range(0,len(original_tragectory)):
    positions =[]
    errors=[]
    for j in range(0,3):
        for k in range(len(candidate_pos[i][j])):
            position = candidate_pos[i][j][k]
            error = 0
            for l in range(0,3):
                error_inter = math.sqrt(((position[0]-node_pos[l][0])**2) + ((position[1]-node_pos[l][1])**2))
                error = error + math.pow((error_inter - dist_i[i][l]),2)
            errors.append(error)
            positions.append(position)
    min_error = min(errors)
    min_index = errors.index(min_error)
    predicted_pos = positions[min_index]

    distance_error.append(dist(predicted_pos[0],predicted_pos[1],original_tragectory[i]))


print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
distcumulativeEror=np.sum(distance_error)
distmeanError=np.average(distance_error)
distStandardDeviationError=np.std(distance_error)
print("DIST_ERROR:   Cumulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))


