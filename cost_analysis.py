#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:52:24 2022

@author: ziva
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import array
import pandas as pd

experiment_numbers = ["858403345", "926647012", "990508730"]
model_names = ["A", "B", "C"]
thresholds = [92, 90, 85, 80]


def compute_when_threshold_acc_is_reached(file, threshold):
    f = open(file, "r")
    skip = False
    check_for_num_steps = True
    n_run = 0
    n_steps_to_threshold = []
    start_id = 0
    for i, l in enumerate(f):
        a, b = l.split("], [")
        a = a[2:].split(", ") # a = [num_run, num_epoch, num_step]
        if check_for_num_steps and int(a[1])==1:
            check_for_num_steps = False
            num_steps_per_epoch = i
        b = float(b[:-3].split(", ")[0])
        if (b >= threshold) and not skip:
            skip = True
            n_steps_to_threshold += [float(i-start_id)]
        if int(a[0]) > n_run:
            if int(a[0]) > len(n_steps_to_threshold):
                n_steps_to_threshold += [float('nan')]    
            skip = False
            start_id = i
            n_run = int(a[0])
    return 1/num_steps_per_epoch * np.array(n_steps_to_threshold)

def return_means_and_stds(cost_vectors):
    means = []
    stds = []
    for i in range(len(cost_vectors)):
        cv = [j for j in cost_vectors[i][1] if j != -1]
        means += [np.max(cv)]
        #means += [np.mean(cv)]
        stds += [np.std(cv)]
    return means, stds

Means = {}
Stds = {}
Epochs = {}

for model_name in model_names:
    print("model_name: ", model_name, " of ", model_names)
    Means[model_name] = {}
    Stds[model_name] = {}
    Epochs[model_name] = {}
    for experiment_number in experiment_numbers:
        Epochs[model_name][experiment_number] = {}
        print("  experiment_number: ", experiment_number, " of ", experiment_numbers)
        cost_file_path = "data/new/yue/" + experiment_number + "_cost_" + model_name + ".log"
        acc_file_path = "data/new/yue/" + experiment_number + "_acc_" + model_name + ".log"
        cost_file = open(cost_file_path, "r")
        exec("cost_vectors = [" + cost_file.read() + "]")
        cost_file.close()
        means, stds = return_means_and_stds(cost_vectors)
        print("    The average mean: ", np.mean(means))
        print("    The average std: ", np.mean(stds))
        Means[model_name][experiment_number] = means
        Stds[model_name][experiment_number] = stds
        for threshold in thresholds:
            Epochs[model_name][experiment_number][threshold] = compute_when_threshold_acc_is_reached(acc_file_path, threshold)
        


#print(Means)


#costA1 = "data/new/yue/858403345_cost_A.log"
#costB1 = "data/new/yue/858403345_cost_B.log"
#costC1 = "data/new/yue/858403345_cost_C.log"
#fA1 = open(costA1, "r")
#fB1 = open(costB1, "r")
#fC1 = open(costC1, "r")
#
#costA2 = "data/new/yue/926647012_cost_A.log"
#costB2 = "data/new/yue/926647012_cost_B.log"
#costC2 = "data/new/yue/926647012_cost_C.log"
#fA2 = open(costA2, "r")
#fB2 = open(costB2, "r")
#fC2 = open(costC2, "r")
#
#costA3 = "data/new/yue/990508730_cost_A.log"
#costB3 = "data/new/yue/990508730_cost_B.log"
#costC3 = "data/new/yue/990508730_cost_C.log"
#fA3 = open(costA3, "r")
#fB3 = open(costB3, "r")
#fC3 = open(costC3, "r")

# Uncomment and adjust if cost D should also be analysed:
# costD = "data/new/yue/858403345_cost_D.log"
# fD = open(costD, "r")

#exec("cost_vectors_A1 = [" + fA1.read() + "]")
#fA1.close()
#
#exec("cost_vectors_B1 = [" + fB1.read() + "]")
#fB1.close()
#
#exec("cost_vectors_C1 = [" + fC1.read() + "]")
#fC1.close()
#
#exec("cost_vectors_A2 = [" + fA2.read() + "]")
#fA2.close()
#
#exec("cost_vectors_B2 = [" + fB2.read() + "]")
#fB2.close()
#
#exec("cost_vectors_C2 = [" + fC2.read() + "]")
#fC2.close()
#
#exec("cost_vectors_A3 = [" + fA3.read() + "]")
#fA3.close()
#
#exec("cost_vectors_B3 = [" + fB3.read() + "]")
#fB3.close()
#
#exec("cost_vectors_C3 = [" + fC3.read() + "]")
#fC3.close()

# Uncomment and adjust if cost D should also be analysed:
#exec("cost_vectors_D = [" + fD.read() + "]")
#fD.close()

#if not len(cost_vectors_A)==len(cost_vectors_B)==len(cost_vectors_C):
#    assert False


#meanA1, stdA1 = return_means_and_stds(cost_vectors_A1)
#meanB1, stdB1 = return_means_and_stds(cost_vectors_B1)
#meanC1, stdC1 = return_means_and_stds(cost_vectors_C1)
##meanD, stdD = return_means_and_stds(cost_vectors_D)
#meanA2, stdA2 = return_means_and_stds(cost_vectors_A2)
#meanB2, stdB2 = return_means_and_stds(cost_vectors_B2)
#meanC2, stdC2 = return_means_and_stds(cost_vectors_C2)
#meanA3, stdA3 = return_means_and_stds(cost_vectors_A3)
#meanB3, stdB3 = return_means_and_stds(cost_vectors_B3)
#meanC3, stdC3 = return_means_and_stds(cost_vectors_C3)
    
#print("Average mean A: ", np.mean(meanA1), np.mean(meanA2), np.mean(meanA3) )
#print("Average mean B: ", np.mean(meanB1), np.mean(meanB2), np.mean(meanB3))
#print("Average mean C: ", np.mean(meanC1), np.mean(meanC2), np.mean(meanC3))
##print("Average mean D: ", np.mean(meanD))
#print("Standard deviation of means of A: ", np.std(meanA1), np.std(meanA2), np.std(meanA3))
#print("Standard deviation of means of B: ", np.std(meanB1), np.std(meanB2), np.std(meanB3))
#print("Standard deviation of means of C: ", np.std(meanC1), np.std(meanC2), np.std(meanC3))
##print("Standard deviation of means of D: ", np.std(meanD))
#print("Average std A: ", np.mean(stdA1), np.mean(stdA2), np.mean(stdA3))
#print("Average std B: ", np.mean(stdB1), np.mean(stdB2), np.mean(stdB3))
#print("Average std C: ", np.mean(stdC1), np.mean(stdC2), np.mean(stdC3))
##print("Average std D: ", np.mean(stdD))