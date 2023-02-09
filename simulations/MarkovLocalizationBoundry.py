import math
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as pb
import random
from datetime import datetime
import time

# distance Calculation
def dist(x, y, pos):
    return math.sqrt((pos[0]-x)**2 + (pos[1]-y)**2)

# Simulation Environment variable Initialization
areaSize=(30, 30)
node_positions = (areaSize[0]+6,areaSize[1]+6)
node_pos=[(-node_positions[0],node_positions[1]),(node_positions[0],node_positions[1]),(node_positions[0],-node_positions[1]),(-node_positions[0],-node_positions[1])] # for 4 nodes
# node_pos=[(-node_positions[0],-node_positions[1]),(-node_positions[0],node_positions[1]),(node_positions[0],-node_positions[1])] # fro 3 nodes
initial_pos=(0,0) 
NOISE_LEVEL=1
RESOLUTION=5
Delta_X = 72
Delta_Y = 72
STEP_SIZE=1/RESOLUTION

# RSSI signal generation at pos(x,y) using path-loos model
def gen_wifi(freq=2.4, power=20, trans_gain=0, recv_gain=0, size=areaSize, pos=(5,5), shadow_dev=2, n=3,noise=NOISE_LEVEL):
    if pos is None:
        pos = (random.randrange(size[0]), random.randrange(size[1]))

    random.seed(datetime.now())
    rss0 = power + trans_gain + recv_gain + 20 * math.log10(3 / (4 * math.pi * freq * 10))
    rss0=rss0-noise*random.random()
    normal_dist = np.random.normal(0, shadow_dev, size=[size[0]+1, size[1]+1])
    rss = []

    random.seed(datetime.now())

    for x in range(0,4):
        distance = dist(node_pos[x][0], node_pos[x][1], pos)
        val =rss0 - 10 * n * math.log10(distance) + normal_dist[int(pos[0])][int(pos[1])]
        rss.append(val-noise*random.random())

    return rss


# Robot path trajectory generation

## To visualize Simulation, uncomment the plt code

# plt.ion()
# plt.title("Robot Trajectory")
# plt.ylim(-areaSize[1],areaSize[1])
# plt.xlim(-areaSize[0],areaSize[0])
# plt.plot( node_pos[0],node_pos[1], 'ro', markersize=5, clip_on=False, zorder=100)
# plt.plot( node_pos[2],node_pos[3], 'ro', markersize=5, clip_on=False, zorder=100)
# rss_pos=[(node_pos[0][0]+3,node_pos[0][1]-1),(node_pos[1][0]-25,node_pos[1][1]-1),(node_pos[2][0]-25,node_pos[2][1]-1),(node_pos[3][0]+3,node_pos[3][1]-1)]
text=[]
overall_rss=[]
original_tragectory=[]
Previous_pos = initial_pos

# for i in range(0,4):
#     text.append(plt.text(rss_pos[i][0],rss_pos[i][1], 'RSS'+str(i), fontsize=10))

def move(pos):
    x,y=pos[0],pos[1]
    original_tragectory.append((x,y))
    # plt.plot([Previous_pos[0],x],[Previous_pos[1],y],'g-',linewidth=2,clip_on=False)
    rss = gen_wifi(pos=(x,y))
    overall_rss.append(rss)
    # for i in range(0,4):
    #     text[i].set_text('RSS'+str(i+1)+'='+str(round(rss[i], 2)))
    # plt.draw()
    # plt.pause(0.0001)
x=0
for y in np.arange(0,areaSize[1],STEP_SIZE):
    move((x,y))
    Previous_pos = (x,y)
y=areaSize[1]
for x in np.arange(0,areaSize[0],STEP_SIZE):
    move((x,y))
    Previous_pos = (x,y)
x=areaSize[0]
for y in np.arange(areaSize[1],-areaSize[1],-STEP_SIZE):
    move((x,y))
    Previous_pos = (x,y)
y=-areaSize[1]
for x in np.arange(areaSize[0],-areaSize[0],-STEP_SIZE):
    move((x,y))
    Previous_pos = (x,y)
x=-areaSize[0]
for y in np.arange(-areaSize[0],areaSize[0],STEP_SIZE):
    move((x,y))
    Previous_pos = (x,y)
y=areaSize[1]
for x in np.arange(-areaSize[0],0,STEP_SIZE):
    move((x,y))
    Previous_pos = (x,y)

# plt.show(block=False)
# plt.savefig('PF_tragectory_boundry.png')
# plt.pause(3)
# plt.clf()
# plt.close()


# plt.ion()
prev =()
doa=[]
# plt.title("Direction of Arrival")
# plt.xlabel("time")
# plt.ylabel("DOA")
for i in range(0,len(overall_rss)):
    inner_curr = i
    limit = i-500 if i>500 else 0
    est_sin_sum = 0
    est_cos_sum = 0
    starting_curr = inner_curr
    weight_sum = 0
    # average estimated DoA calculated
    while inner_curr >= limit:
    # print(str(overall_rss[i]))
        gx = ((overall_rss[i][1]-overall_rss[i][0])/2) + ((overall_rss[i][2]-overall_rss[i][3])/2)
        gy = ((overall_rss[i][1]-overall_rss[i][2])/2) + ((overall_rss[i][0]-overall_rss[i][3])/2)
        # gx = ((overall_rss[i][2]-overall_rss[i][0])/Delta_X)
        # gy = ((overall_rss[i][1]-overall_rss[i][0])/Delta_Y)
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
    if not prev:
        prev = (i,avg_grad)
    # print(str(gx)+"\t"+str(gy))
    # plt.plot([prev[0],i],[prev[1],avg_grad],'r-')
    prev=(i,avg_grad)
    # plt.axis('tight')
    # plt.draw()
    # plt.pause(0.0001)
    
# plt.show(block=False)
# plt.pause(3)
# plt.savefig('PF_doa_boundry.png')
# plt.clf()
# plt.close()

# plt.ion()


# plt.ylim(-areaSize[1],areaSize[1])
# plt.xlim(-areaSize[0],areaSize[0])

prev=()
random.seed(datetime.now())
previous_errors =[]
distance_error =[]
particles = []
times = []
Previous_pos = initial_pos
start_time = time.time()

for x in np.arange(-areaSize[0],areaSize[0],STEP_SIZE*2):
    for y in np.arange(-areaSize[1],areaSize[1],STEP_SIZE*2):
        particles.append((x,y))


# plt.title("Predicted Robot Positions")
for i in range(0,len(original_tragectory)):
    positions =[]
    errors=[]
    error=0
    for particle in particles:
        x,y=particle[0],particle[1]
        # print(str(particle))
        actual_rss = gen_wifi(pos=(x,y),noise=0)
        gx = ((actual_rss[1]-actual_rss[0])/2) + ((actual_rss[2]-actual_rss[3])/2)  #for 4 nodes
        gy = ((actual_rss[1]-actual_rss[2])/2) + ((actual_rss[0]-actual_rss[3])/2)
        # gx = ((actual_rss[2]-actual_rss[0])/Delta_X)
        # gy = ((actual_rss[1]-actual_rss[0])/Delta_Y)
        adoa=np.arctan(gy/gx) if gx !=0 else  0
        error=abs(adoa-doa[i])
        positions.append((x,y))
        errors.append(error)


    min_error = min(errors)
    min_index = errors.index(min_error)
    pos=positions[min_index](max_weight)
    pos=positions[max_index]
    previous_errors.append(errors[min_index])
    distance_error.append(dist(pos[0],pos[1],original_tragectory[i]))

# plt.show(block=False)
# plt.savefig('PF_predicted_trajectory_boundry.png')
print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
distcumulativeEror=np.sum(distance_error)/10
distmeanError=np.average(distance_error)/10
distStandardDeviationError=np.std(distance_error)/10
print("DIST_ERROR:   Cumulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))
resultFile = open("error_boundry_Markov_full(6by6).csv", "a")  # append mode
resultFile.write(str(distcumulativeEror)+","+str(distmeanError)+","+str(distStandardDeviationError)+"\n")


resultFile.close()
# plt.pause(3)
# plt.clf()
# plt.close()

