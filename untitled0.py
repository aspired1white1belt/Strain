#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:48:22 2026

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import strain_sim as ss
import time
start_time = time.time()
L = 40+39
steps = 8000
stepsize = L / (steps-1)
fitting_radius = int(0.1/stepsize) #less than 0.25um from the real data point
penalty = 10
def chi_squared(xdata, ydata, yerrdata, xdata_sim, ydata_sim):
    #data_sim has higher sampling rate
    chi_squared = 0
    n = len(xdata)
    for i in range(n):
        j = int(steps * xdata[i] / L) #index in data_sim
        if j-fitting_radius < 0:
            start = 0
            end = j+fitting_radius
        elif j+fitting_radius > (steps - 1):
            start = j-fitting_radius
            end = steps-1
        else:
            start = j-fitting_radius
            end = j+fitting_radius
        for k in range(start, end):
            if xdata[i] < 11 or xdata[i] > 70:
                chi_squared += ((ydata_sim[k] - ydata[i]) / yerrdata[i])**2 
            else:
                chi_squared += penalty*((ydata_sim[k] - ydata[i]) / yerrdata[i])**2

    #reduced_chi_squared = chi_squared / (len(ydata6) - 7)
    return chi_squared

data = np.loadtxt("curated_data.csv", dtype=float, delimiter=",", skiprows=1)
data_x = data[:,0] + 39
data_y = data[:,1]
data_yerr = data[:,2]

A_fitting = [-2]#np.arange(-0.5, -10, -1) #
sigma_fitting = [0.4]#np.arange(0.35, 0.41, 0.01)########
l_fitting =  [4]##np.arange(1, 10, 1)
a = 100000

optimal_params = np.zeros(3)
for i in range(len(A_fitting)):
    for j in range(len(sigma_fitting)):
        for k in range(len(l_fitting)):
            sim_x, sim_y = ss.data_fitting(fitting_param = [A_fitting[i], sigma_fitting[j], l_fitting[k]], discrete_steps=steps)
            # fig, ax = plt.subplots()
            # ax.errorbar(data_x, data_y, yerr=data_yerr, fmt="ko", capsize=6)
            # ax.plot(sim_x, sim_y)
            #plt.show()
            b = chi_squared(data_x, data_y,data_yerr, sim_x, sim_y)
            if b < a:
                a = b
                optimal_params[0] = A_fitting[i]
                optimal_params[1] = sigma_fitting[j]
                optimal_params[2] = l_fitting[k]
  
sim_x, sim_y = ss.data_fitting(fitting_param = optimal_params, discrete_steps=steps)
fig, ax = plt.subplots()
ax.errorbar(data_x, data_y, yerr=data_yerr, fmt="ko", capsize=6)
ax.plot(sim_x, sim_y)
plt.show()
print(f"optimal chi_sqaured = {a:3f}")
print(f"optimal parameters: A = {optimal_params[0]}, sigma = {optimal_params[1]}, l = {optimal_params[2]}")
print(f"runtime = {time.time() - start_time}s")
data_to_save = np.zeros((steps, 2))
data_to_save[:,0] = sim_x[:,0] - 39
data_to_save[:,1] = sim_y
np.savetxt("strain_simulation.csv", data_to_save, delimiter=",")
