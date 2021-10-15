import math
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as pb
from matplotlib.lines import Line2D

import random
from datetime import datetime
import time


def dist(x, y, pos):
    return math.sqrt((pos[0]-x)**2 + (pos[1]-y)**2)




areaSize=(20, 20)
node_positions = (areaSize[0]+6,areaSize[1]+6)
node_pos=[(-node_positions[0],node_positions[1]),(node_positions[0],node_positions[1]),(node_positions[0],-node_positions[1]),(-node_positions[0],-node_positions[1])]
initial_pos=(0,0)
possible_value = list(range(-20, 20))
num_particles = 1000
NOISE_LEVEL=1
RESOLUTION=10
STEP_SIZE=1/RESOLUTION
n = 3
random.seed(datetime.now())
original_rss0 = 20 + 20 * math.log10(3 / (4 * math.pi * 2.4 * 10))
rss0 = original_rss0 - NOISE_LEVEL*random.random()#random.randint(NOISE_LEVEL,NOISE_LEVEL+10)


def gen_wifi(freq=2.4, power=20, trans_gain=0, recv_gain=0, size=areaSize, pos=(5,5), shadow_dev=2, n=3,noise=NOISE_LEVEL):
    if pos is None:
        pos = (random.randrange(size[0]), random.randrange(size[1]))

    normal_dist = np.random.normal(0, shadow_dev, size=[size[0]+1, size[1]+1])
    rss = []

    random.seed(datetime.now())

    for x in range(0,4):
        distance = dist(node_pos[x][0], node_pos[x][1], pos)
        val =rss0 - 10 * n * math.log10(distance) + normal_dist[int(pos[0])][int(pos[1])]
        rss.append(val-noise*random.random())

    return rss



# plt.ion()
plt.figure(figsize=(10, 10))
plt.title("Node Position Prediction",fontsize=25)
plt.ylim(-areaSize[1],areaSize[1])
plt.xlim(-areaSize[0],areaSize[0])
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
# plt.plot( node_pos[0],node_pos[1], 'ro', markersize=5, clip_on=False, zorder=100)
# plt.plot( node_pos[2],node_pos[3], 'ro', markersize=5, clip_on=False, zorder=100)

rss_pos=[(node_pos[0][0]+2,node_pos[0][1]),(node_pos[1][0]-5,node_pos[1][1]),(node_pos[2][0]-5,node_pos[2][1]),(node_pos[3][0]+2,node_pos[3][1])]
text=[]
overall_rss=[]
original_tragectory=[]
Previous_pos = initial_pos

# for i in range(0,4):
#     text.append(plt.text(rss_pos[i][0],rss_pos[i][1], 'Node '+str(i), fontsize=10))

for i in range(10):
    random.seed(datetime.now())
    x_pos = random.randint(-areaSize[0]+2,areaSize[0]-2)
    random.seed(datetime.now())
    y_pos = random.randint(-areaSize[1]+2,areaSize[1]-10)
    original_tragectory.append((x_pos,y_pos))
    inter_rss = []
    for j in range(100):
        inter_rss.append(gen_wifi(pos=(x_pos,y_pos),noise=2))
    overall_rss.append(np.array(np.mean(inter_rss, axis=0)))

   
plt.scatter(np.array(original_tragectory)[:,0],np.array(original_tragectory)[:,1] , marker='o',s=100, color='black',label='Actual Device Location')
plt.draw()

def calc_dist(rss):
    cal_d= pow(10,((rss-rss0)/(-10*n)))
    return cal_d

######################### Weighted Centroid #########################

def wcl(weight,x,y):
    xiwi=np.multiply(x,weight)
    yiwi=np.multiply(y,weight)
    xw=np.sum(xiwi)/np.sum(weight)
    yw=np.sum(yiwi)/np.sum(weight)
    return xw,yw

distance_error =[]
start_time = time.time()
WCPosistions = []
for i in range(0,len(original_tragectory)):
    weight_arr = []
    x = overall_rss[i]
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    cal_d=calc_dist(x1)
    weight_arr=np.append(weight_arr,(1/cal_d))
    cal_d=calc_dist(x2)
    weight_arr=np.append(weight_arr,(1/cal_d))
    cal_d=calc_dist(x3)
    weight_arr=np.append(weight_arr,(1/cal_d))
    cal_d=calc_dist(x4)
    weight_arr=np.append(weight_arr,(1/cal_d))
    res_x,res_y=wcl(weight_arr,[node_pos[j][0] for j in range(0,4)],[node_pos[j][1] for j in range(0,4)]) 
    WCPosistions.append((res_x,res_y))
    distance_error.append(dist(res_x,res_y,original_tragectory[i]))
plt.scatter(np.array(WCPosistions)[:,0],np.array(WCPosistions)[:,1],marker='s',s=80, color='g',label='Weighted Centroid')
plt.draw()

print("--------Statistics for Weighted Centroid-------------")
print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
distcumulativeEror=np.sum(distance_error)
distmeanError=np.average(distance_error)
distStandardDeviationError=np.std(distance_error)
# text.append(plt.text(8,28, 'Root Mean Square Error', fontsize=13))
# text.append(plt.text(4,26, 'Weighted Centroid :  '+"{:.2f} m".format(distmeanError), fontsize=15))
print("DIST_ERROR:   Cumulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))

######################### Non Linear Least Square Error Minimization #########################

dist_i = []
candidate_pos = []
for i in range(0,len(overall_rss)):
    dist_j = []
    for j in range(0,4):
        dist_j.append(calc_dist(overall_rss[i][j]))
    dist_i.append(dist_j)
    # print(dist_j)
    candidate_pos_j =[]
    for j in range(0,4):
        y = node_pos[j][1]-dist_j[j]
        x = node_pos[j][0]
        # print(x,y)
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
LSE_positions = []
start_time = time.time()


for i in range(0,len(original_tragectory)):
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
    LSE_positions.append(predicted_pos)
    distance_error.append(dist(predicted_pos[0],predicted_pos[1],original_tragectory[i]))

plt.scatter(np.array(LSE_positions)[:,0],np.array(LSE_positions)[:,1],marker='^',s=90, color='r',label='Non Linear Least Square Error')
plt.draw()
print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
distcumulativeEror=np.sum(distance_error)
distmeanError=np.average(distance_error)
distStandardDeviationError=np.std(distance_error)
# text.append(plt.text(4,24, 'Non Linear Least Square Error :  '+"{:.2f} m".format(distmeanError), fontsize=15))
print("DIST_ERROR:   Cumulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))

######################### Differential RSS #########################


previous_errors =[]
distance_error =[]
particles = []
times = []
differential_positions = []
start_time = time.time()

for x in range(num_particles):
    random.seed(datetime.now())
    particles.append((random.choice(possible_value),random.choice(possible_value)))
for i in range(0,len(original_tragectory)):
    positions =[]
    errors=[]
    weights =[]
    rands = []
    range_probs = []
    error=0
    for particle in particles:
        x,y=particle[0],particle[1]
        actual_rss = gen_wifi(pos=(x,y),noise=0)
        error=np.sum(np.subtract(actual_rss,overall_rss[i]))
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
    differential_positions.append(pos)
    distance_error.append(dist(pos[0],pos[1],original_tragectory[i]))

plt.scatter(np.array(differential_positions)[:,0],np.array(differential_positions)[:,1],marker='x',s=80, color='gray',label='Grid-Based Differential RSS')
plt.draw()

print("-------- Statistics for Grid-Based Differential RSS -------------")
print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
distcumulativeEror=np.sum(distance_error)
distmeanError=np.average(distance_error)
distStandardDeviationError=np.std(distance_error)
# text.append(plt.text(4,22, 'Grid-Based Differential RSS :  '+"{:.2f} m".format(distmeanError), fontsize=15))
print("DIST_ERROR:   Cumulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))






doa=[]
for i in range(0,len(overall_rss)):
    inner_curr = i
    limit = i-500 if i>500 else 0
    est_sin_sum = 0
    est_cos_sum = 0
    starting_curr = inner_curr
    weight_sum = 0
    while inner_curr >= limit:
        gx = ((overall_rss[i][1]-overall_rss[i][0])/2) + ((overall_rss[i][2]-overall_rss[i][3])/2)
        gy = ((overall_rss[i][1]-overall_rss[i][2])/2) + ((overall_rss[i][0]-overall_rss[i][3])/2)
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

######################### Markov Approach #########################


previous_errors =[]
distance_error =[]
particles = []
times = []
PF_positions = []
start_time = time.time()

# for x in range(num_particles):
#     random.seed(datetime.now())
#     particles.append((random.choice(possible_value),random.choice(possible_value)))
for x in np.arange(-areaSize[0],areaSize[0],2):
    for y in np.arange(-areaSize[1],areaSize[1],2):
        particles.append((x,y))
for i in range(0,len(original_tragectory)):
    positions =[]
    errors=[]
    weights =[]
    rands = []
    range_probs = []
    error=0
    for particle in particles:
        x,y=particle[0],particle[1]
        actual_rss = gen_wifi(pos=(x,y),noise=0)
        gx = ((actual_rss[1]-actual_rss[0])/2) + ((actual_rss[2]-actual_rss[3])/2)
        gy = ((actual_rss[1]-actual_rss[2])/2) + ((actual_rss[0]-actual_rss[3])/2)
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
    PF_positions.append(pos)
    distance_error.append(dist(pos[0],pos[1],original_tragectory[i]))

plt.scatter(np.array(PF_positions)[:,0],np.array(PF_positions)[:,1],marker='+',s=80, color='y',label='Markov Grid')
plt.draw()
print("-------- Statistics for Markov Grid Localization -------------")
print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
distcumulativeEror=np.sum(distance_error)
distmeanError=np.average(distance_error)
distStandardDeviationError=np.std(distance_error)
# text.append(plt.text(4,20, 'Proposed Approach :  '+"{:.2f} m".format(distmeanError), fontsize=15))
print("DIST_ERROR:   Cumulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))


######################### PF-Based Proposed Approach #########################





previous_errors =[]
distance_error =[]
particles = []
times = []
PF_positions = []
start_time = time.time()

for x in range(num_particles):
    random.seed(datetime.now())
    particles.append((random.choice(possible_value),random.choice(possible_value)))
for i in range(0,len(original_tragectory)):
    positions =[]
    errors=[]
    weights =[]
    rands = []
    range_probs = []
    error=0
    for particle in particles:
        x,y=particle[0],particle[1]
        actual_rss = gen_wifi(pos=(x,y),noise=0)
        gx = ((actual_rss[1]-actual_rss[0])/2) + ((actual_rss[2]-actual_rss[3])/2)
        gy = ((actual_rss[1]-actual_rss[2])/2) + ((actual_rss[0]-actual_rss[3])/2)
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
    PF_positions.append(pos)
    distance_error.append(dist(pos[0],pos[1],original_tragectory[i]))

plt.scatter(np.array(PF_positions)[:,0],np.array(PF_positions)[:,1],marker='*',s=80, color='y',label='Proposed PF-DOA Approach (with odom)')
plt.draw()
print("-------- Statistics for DOA Based PF Localization -------------")
print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
distcumulativeEror=np.sum(distance_error)
distmeanError=np.average(distance_error)
distStandardDeviationError=np.std(distance_error)
# text.append(plt.text(4,20, 'Proposed Approach :  '+"{:.2f} m".format(distmeanError), fontsize=15))
print("DIST_ERROR:   Cumulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))

plt.legend(loc="upper left" , prop={'size':20})

plt.show(block=False)
plt.ioff()
plt.show()


