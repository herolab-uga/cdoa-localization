# Device Localization Using Particle Filter over DOA of Wireless Signals
## Overview
This repository contains the datasets and scripts used in the experimentation for localization algorithms. Three set of experimentational content is provided in this repository: 1. Simulation, 2.Dataset1, and 3. Dataset2. Each content in enclosed in separate folder with the respective name. Simulation scripts generate data internally, However, realworld exprimentation require data files which are included with the respective folder in the repository.
Visualize the system as: 

![Overview](/images/overview.png)
### Simulation
Simulation experimentations taken place for 6 x 6 meter of bounded region where Access Point is placed on the top of the moving robot and wireless sensor nodes places at the fixed positions (corners) which are (0,0), (0,6), (6,0), and (6,6) respectively. Robot followed three different trajectoriers: Inside, Boundary and Diagonal. For each of the trajectory we predicted the position using multiple localization techniques and proposed approach as well.

![Combined Trajectory](/images/combined_trajectories.png).

In the Simulation folder there is a separate python script file for each localization technique.

To execute the script run: `$ python3 <script_file>`
### Dataset1
It contains script files same as simulation to run algorithms on data available in csv files.

To execute the script run: `$ python3 <script_file> <dataset file>`
 
Experimenatation Testbed and datapoints can be visualized as:

![dataset1](/images/dataset1.png)

The reference at [1] povides details of the dataset completly.
 
 ### Dataset2
It contains script files same as simulation to run algorithms on data available in csv files.

To execute the script run: `$ python3 <script_file> <dataset file>`

Experimenatation Testbed and datapoints can be visualized as:

![dataset2](/images/dataset2.png)

The reference at [2] povides details of the dataset completly.
### References

[1] "Rssi  dataset  for  indoor  localization  fingerprinting."  [Online].Available: https://github.com/pspachos/RSSI-Dataset-for-Indoor-Localization-Fingerprinting.git

[2]  P.    Szeli ́nski    and    M.    Nikodem,    "Multi-channel    ble    rssimeasurements for indoor localization," 2021. [Online]. Available: https://dx.doi.org/10.21227/hc74-3s30

