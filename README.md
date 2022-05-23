# Device Localization Using Particle Filter over DOA of Wireless Signals

## Paper Link
Please check the paper's preprint and Appendix at http://hero.uga.edu/research/localization

## Overview
This repository contains the datasets and scripts used in the experimentation for localization algorithm using Wireless signal Direction of Arrival (DOA) estimated using a mobile robot - wireless sensor network (WSN) cooperation mechanism in real-time. Three set of experimentational content is provided in this repository: 1. Simulation, 2.Dataset1, 3. Dataset2, 4. Hardware. Each content in enclosed in separate folder with the respective name. Simulation scripts generate data internally, However, realworld exprimentation require data files which are included with the respective folder in the repository.

The proposed localization framwwork can be visualizes as: 

![Overview](/images/overview.png)

On the Left, the Wireless Sensor Network (WSN) anchor nodes are placed at the corners and the Mobile Robot can be localized within the WSN's boundary polygon. 

On the Right, the Access Points (AP) are placed at the corers and the mobile robot can be localized within thie AP boundaries.

Although we present the experimental results for the WSN nodes/AP anchors in a specific square/triangular shape, the approach can be generalized to other configurations as long as all the nodes are not colinear.

A hardware demonstration experiment setup is shown below using the configuration of WSN anchor nodes.

![Hardware Demo](/images/hardware_testbed.png)


Below, we present different datasets and associated robot localization trajectories or information.


### Dataset 0 [Simulation Experiments Datasets]

Simulation experimentations taken place for 6 x 6 meter of bounded region where Access Point is placed on the top of the moving robot and wireless sensor nodes places at the fixed positions (corners) which are (0,0), (0,6), (6,0), and (6,6) respectively. Robot followed three different trajectoriers: Inside, Boundary and Diagonal. For each of the trajectory we predicted the position using multiple localization techniques and proposed approach as well.

![Combined Trajectory](/images/combined_trajectories.png).

In the Simulation folder there is a separate python script file for each localization technique.

To execute the script run: `$ python3 <script_file>`

### Dataset1 [Real-world Public Dataset]

It contains script files same as simulation to run algorithms on data available in csv files.

To execute the script run: `$ python3 <script_file> <dataset file>`
 
Experimenatation Testbed and datapoints can be visualized as:

![dataset1](/images/dataset1.png)

The reference at [1] povides details of the dataset completely.
 
 ### Dataset2 [Real-world Public Dataset]

It contains script files same as simulation to run algorithms on data available in csv files.

To execute the script run: `$ python3 <script_file> <dataset file>`

Experimenatation Testbed and datapoints can be visualized as:

![dataset2](/images/dataset2.png)
The reference at [2] povides details of the dataset completely.

 ### Hardware [Hardware Demonstration Experiment Dataset]
 
This folder contains ros package (ros_network_analysis) which consists of wireless publisher node scripts along with the server subscriber script.

To launch the network_analysis run: `$ roslaunch network_analysis wireless_quality.launch <node_id>`

Another ros package for ROS master named ros_rssi_collaboration needs to be run to receive rssi from each node in a synchronous way.

To launch the rssi_collaboration run: `$ roslaunch rssi_collaboration rssi_collaboration.launch`

Furthemore, a folder contains dataset in the form of rosbags, one can easily extract data rosbags. However, we also have provided dataset int he form of .csv for convinience in the respective folders.
It also contains script file to run algorithms on data available in csv files.

To execute the script run: `$ python3 <script_file> <dataset file>`

Experimenatation Testbed and datapoints can be visualized as:

![trajectories](/hardware/trajectories/hardware_experiment_trajectory.png)

### References

[1] Pachos et al. "Rssi  dataset  for  indoor  localization  fingerprinting."  [Online]. Available: https://github.com/pspachos/RSSI-Dataset-for-Indoor-Localization-Fingerprinting.git

[2]  P.    Szeli ́nski    and    M.    Nikodem,    "Multi-channel    BLE    RSSI measurements for indoor localization," 2021. [Online]. Available: https://dx.doi.org/10.21227/hc74-3s30

