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
initial_pos=(0,0)
possible_x = list(range(1, 39))
possible_y = list(range(1, 39))
num_particles = 200
NOISE_LEVEL=1
RESOLUTION=10
STEP_SIZE=1/RESOLUTION
if 'wifi' in sys.argv[1]:
    rss0 = -45.73
    pathloss_exponent = 2.162
elif 'ble' in sys.argv[1]:
    rss0 = -75.48
    pathloss_exponent = 2.271 
elif 'zigbee' in sys.argv[1]:
    rss0 = -50.33
    pathloss_exponent = 2.935
def gen_wifi(freq=2.4, power=20, trans_gain=0, recv_gain=0, size=areaSize, pos=(5,5), shadow_dev=1, n=pathloss_exponent,noise=NOISE_LEVEL):
    if pos is None:
        pos = (random.randrange(size[0]), random.randrange(size[1]))

    random.seed(datetime.now())

    normal_dist = np.random.normal(0, shadow_dev, size=[size[0]+1, size[1]+1])
    rss = []

    random.seed(datetime.now())

    for x in range(0,3):
        distance = dist(node_pos[x][0], node_pos[x][1], pos)
        val =rss0 - 10 * n * math.log10(distance) + normal_dist[int(pos[0])][int(pos[1])]
        rss.append(val-noise*random.random())

    return rss


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


doa=[]
for i in range(0,len(overall_rss)):
    inner_curr = i
    limit = i-500 if i>500 else 0
    est_sin_sum = 0
    est_cos_sum = 0
    starting_curr = inner_curr
    weight_sum = 0
    # average estimated DoA calculated
    while inner_curr >= limit:

        gx = ((overall_rss[i][2]-overall_rss[i][0])/4)
        gy = ((overall_rss[i][1]-overall_rss[i][0])/4)

        estimated_grad=np.arctan(gy/gx)
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


random.seed(datetime.now())
previous_errors =[]
distance_error =[]
times = []
Previous_pos = initial_pos
start_time = time.time()

for i in range(0,len(original_tragectory)):
    particles = []
    for k in range(num_particles):
        particles.append((random.choice(possible_x)/10,random.choice(possible_y)/10))
    positions =[]
    errors=[]
    weights =[]
    rands = []
    range_probs = []
    error=0
    for particle in particles:
        x,y=particle[0],particle[1]
        actual_rss = gen_wifi(pos=(x,y),noise=0)
        gx = ((actual_rss[2]-actual_rss[0])/4)
        gy = ((actual_rss[1]-actual_rss[0])/4)
        adoa=np.arctan(gy/gx) if gx !=0 else  0
        error=abs(adoa-doa[i])
        if previous_errors:
            std_error=np.std(previous_errors)
        else:
            std_error=0.001
        
        std_error=np.std(np.subtract(actual_rss,overall_rss[i]))
        omega=((1/((4*std_error)*math.sqrt(2*math.pi)))*(math.pow(math.e,-(math.pow(error,2)/2*np.square(4*std_error)))))
        for j in range(len(previous_errors)-1,len(previous_errors)-4 if len(previous_errors) > 5 else 0,-1):
            omega=omega*((1/((4*std_error)*math.sqrt(2*math.pi)))*(math.pow(math.e,-(math.pow(previous_errors[j],2)/2*np.square(4*std_error)))))
    
        weights.append(omega)
        positions.append((x,y))
        errors.append(error)
        
    sum_weight=np.sum(weights)
    if sum_weight == 0:
        pass
    for j in range(0,len(weights)):
        weights[j]=weights[j]/sum_weight


    max_weight = max(weights)
    max_index = weights.index(max_weight)
    pos=positions[max_index]
    previous_errors.append(errors[max_index])
    distance_error.append(dist(pos[0],pos[1],original_tragectory[i]))


print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
rsscumulativeEror=np.sum(previous_errors)
rssmeanError=np.average(previous_errors)
rssStandardDeviationError=np.std(previous_errors)
distcumulativeEror=np.sum(distance_error)
distmeanError=np.average(distance_error)
distStandardDeviationError=np.std(distance_error)
print("DIST_ERROR:   Cummulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))

