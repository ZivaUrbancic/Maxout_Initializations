#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:46:46 2022

@author: ziva
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import array
import pandas as pd

alpha = 80 #threshold
accA = "data/new/yue/858403345_acc_A.log"
accB = "data/new/yue/858403345_acc_B.log"
accC = "data/new/yue/858403345_acc_C.log"
#accD = "data/new/yue/858403345_acc_D.log"

def compute_when_threshold_acc_is_reached(file, threshold):
    '''
    Analyse a log file of an experiment containing accuracies to compute
    the rational values enoding the epochs (and steps) of when the threshold 
    accuracy is reached in each run. Returns an array with one rational value 
    per run.
    
    Parameters
    ----------
    file : string
        Path to file containing cost vectors.
    threshold : int or float
        Threshold accuracy.
    
    Returns
    -------
    np.array(float)
        The points (with unit being 1 epoch) in training when threshold
        accuracy was reached for each run of experiment.

    '''
    
    f = open(file, "r")
    skip = False                 # to flag if we want to skip an entry
    check_for_num_steps = True   # are we checking for how many steps per epoch
    n_run = 0                    # counter of the run number
    n_steps_to_threshold = []    # init list of ids where threshold was reached
    start_id = 0                 # save the row index where a new run begins
    
    # Loop through the lines in the file:
    for i, l in enumerate(f):
        a, b = l.split("], [")
        a = a[2:].split(", ")              # a = [num_run, num_epoch, num_step]
        b = float(b[:-3].split(", ")[0])   # b = overall accuracy
        
        # If you reached the beginning of the second epoch (of the first run):
        if check_for_num_steps and int(a[1])==1:
            check_for_num_steps = False    # stop checking for number of steps
            num_steps_per_epoch = i        # and save it in num_steps_per_epoch
        
        # If the accuracy is above the threshold and we are not skipping entries:
        if (b >= threshold) and not skip:
            skip = True        # skip all the other entries for the run
            n_steps_to_threshold += [float(i-start_id)] # apend current id
            
        # If this is the first entry of a new run
        if int(a[0]) > n_run:
            # Check if previous run reached the threshold accuracy
            if int(a[0]) > len(n_steps_to_threshold):
                n_steps_to_threshold += [float('nan')] # append nan if not
            skip = False       # in either case, stop skipping entries
            start_id = i       # remember the id as start of new run
            n_run = int(a[0])  # adjust the counter of runs
    return 1/num_steps_per_epoch * np.array(n_steps_to_threshold)



A = compute_when_threshold_acc_is_reached(accA, alpha)
B = compute_when_threshold_acc_is_reached(accB, alpha)
C = compute_when_threshold_acc_is_reached(accC, alpha)
#D = compute_when_threshold_acc_is_reached(accD, alpha)
print(len(A))

print("Threshold ", alpha, " reached after: ")
print("  ", np.mean(A), " epochs for A with std ", np.std(A))
print("  ", np.mean(B), " epochs for B with std ", np.std(B))
print("  ", np.mean(C), " epochs for C with std ", np.std(C))
#print("     ", np.mean(D), " epochs for D with std ", np.std(D))
