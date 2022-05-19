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






# pos= (4,7)
# pos = (1,14)
# Next Pos(-5,-1)
#freq GHz
#power in decibels per milliwatt (dBm)
areaSize=(2.34, 1.75)
#(x,-y)(-x,-y),(-x,y),(x,y)
#(-150,93),(150,93),(150,-93),(-150,-93)
# For Inner nodes
#node_pos=[(-2.40,1.20),(1.10,1.80),(1.80,-1.70),(-1.70,-2.30)]
node_pos=[(0,0),(0,1.75),(2.34,1.75),(2.34,0)]
initial_pos=(1.199,.0898)
possible_x = list(np.arange(0.9,1.3,0.05))
possible_y = list(np.arange(0.6,1,0.05))
num_particles = 1000
NOISE_LEVEL=2
RESOLUTION=1
STEP_SIZE=1/RESOLUTION
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
rss0 = rss0-NOISE_LEVEL*random.random()
print(rss0)

def gen_wifi(freq=2.4, power=20, trans_gain=0, recv_gain=0, size=areaSize, pos=(5,5), shadow_dev=2, n=3,noise=NOISE_LEVEL):
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


# plt.ion()
# plt.title("Robot Trajectory")
# plt.ylim(-areaSize[1],areaSize[1])
# plt.xlim(-areaSize[0],areaSize[0])
# plt.plot( node_pos[0],node_pos[1], 'ro', markersize=5, clip_on=False, zorder=100)
# plt.plot( node_pos[2],node_pos[3], 'ro', markersize=5, clip_on=False, zorder=100)
# rss_pos=[(node_pos[0][0]+3,node_pos[0][1]-1),(node_pos[1][0]-25,node_pos[1][1]-1),(node_pos[2][0]-25,node_pos[2][1]-1),(node_pos[3][0]+3,node_pos[3][1]-1)]
with open('./experiment_bags/corner_exp/extracted_files/combined_sync_data_'+ sys.argv[1] +'.csv') as f:
    dict_from_csv = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
text=[]
overall_rss=[]
original_tragectory=[]
velocity=[]
Previous_pos = initial_pos
for i in range(5,len(dict_from_csv)-5):
    dict=dict_from_csv[i]
    x , y , w , z = float(dict['robot_pos_x']) , float(dict['robot_pos_y']) ,float(dict['robot_pos_w']), 0
    original_tragectory.append((x,y,w,z))
    # plt.plot(dict['robot_pos_x'],dict['robot_pos_y'],'go',markersize=3,clip_on=False)
    # plt.plot([Previous_pos[0],x],[Previous_pos[1],y],'g-',linewidth=2,clip_on=False)
    rss = [np.average([int(float(x['rssi1'])) for x in dict_from_csv[i-5:i+5]]),
            np.average([int(float(x['rssi2'])) for x in dict_from_csv[i-5:i+5]]),
            np.average([int(float(x['rssi3'])) for x in dict_from_csv[i-5:i+5]]),
            np.average([int(float(x['rssi4'])) for x in dict_from_csv[i-5:i+5]])]
    overall_rss.append(rss)
    velocity.append([float(dict['vel_x']),float(dict['vel_w'])])
    # plt.draw()
    # plt.pause(0.0001)
    Previous_pos = (x,y)

# plt.show(block=False)
# plt.savefig('localization_trajectory.png')
# plt.pause(3)
# plt.clf()
# plt.close()

# plt.ion()
# prev =()
# doa=[]
# plt.title("Robot Motion Model")
# plt.xlabel("x")
# plt.ylabel("y")
# for i in range(0,len(original_tragectory)):
#     xd = motion_model(original_tragectory[i],velocity[i])
#     plt.plot([Previous_pos[0],xd[0]],[Previous_pos[1],xd[1]],'g-',linewidth=2,clip_on=False)
#     plt.draw()
#     plt.pause(0.0001)
#     Previous_pos = (xd[0],xd[1])

# plt.show(block=False)
# plt.savefig('model_model_path.png')
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
    if not prev:
        prev = (i,avg_grad)
    # print(str(gx)+"\t"+str(gy))
    # plt.plot([prev[0],i],[prev[1],avg_grad],'r-')
    # prev=(i,avg_grad)
    # plt.axis('tight')
    # plt.draw()
    # plt.pause(0.0001)
    
# plt.show(block=False)
# plt.pause(3)
# plt.savefig('PF_doa_boundry.png')
# plt.clf()
# plt.close()
plt.ion()


# # plt.ylim(-60,60)
# # plt.xlim(-60,60)

prev=()
random.seed(datetime.now())
previous_errors =[]
distance_error =[]
particles = []
times = []
Previous_pos = initial_pos
start_time = time.time()
# for x in np.arange(-areaSize[0],areaSize[0],0.05):
#     for y in np.arange(-areaSize[1],areaSize[1],0.05):
#         particles.append((x,y))
for x in range(num_particles):
    particles.append((random.choice(possible_x),random.choice(possible_y)))
fig, plt_pos = plt.subplots(1,1)
plt_pos.set_title("Trajectory "+sys.argv[1])
# plt_error.set_title('Error Plot')
for i in range(0,len(original_tragectory)):
    positions =[]
    errors=[]
    weights =[]
    rands = []
    range_probs = []
    actual_rss_ls=[]
    error=0
    for particle in particles:
        x,y=particle[0],particle[1]
        actual_rss = gen_wifi(pos=(x,y),noise=0)
        gy = ((actual_rss[1]-actual_rss[0])/2) + ((actual_rss[2]-actual_rss[3])/2)
        gx = ((actual_rss[2]-actual_rss[1])/2) + ((actual_rss[3]-actual_rss[0])/2)
        adoa=np.arctan(gy/gx) if gx !=0 else 0
        error=abs(adoa-doa[i])
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

