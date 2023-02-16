#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:52:24 2022

@author: ziva
"""

# This script contains code for the analysis of correlation between costs of
# regions and rate of convergence.

import numpy as np
#import matplotlib.pyplot as plt
from numpy import array
#import pandas as pd

experiment_numbers = ["858403345", "926647012", "990508730"]
model_names = ["A", "B", "C"]
thresholds = [92, 90, 85, 80]


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

def return_means_and_stds(cost_vectors):    
    '''
    Analyse a log file of an experiment containing cost vectors to compute
    the mean and standard deviation of the costs of regions for each run.
    
    Parameters
    ----------
    cost_vectors : np.array(np.array(float))
        An array of a cost vector for each run.
    
    Returns
    -------
    list(np.float64)
        The means of regions costs for each run.
    list(np.float64)
        The standard deviations of regions costs for each run.

    '''
    
    means = []
    stds = []
    for i in range(len(cost_vectors)):
        cv = [j for j in cost_vectors[i][1] if j != -1] # remove the -1 tags
        means += [np.max(cv)]
        #means += [np.mean(cv)]
        stds += [np.std(cv)]
    return means, stds

# Initialise dictionaries in which to save means, stds and epoch computed with
# above methods.
Means = {}
Stds = {}
Epochs = {}


for model_name in model_names:
    print("model_name: ", model_name, " of ", model_names)
    # Initialise subdictionaries for each model type
    Means[model_name] = {}
    Stds[model_name] = {}
    Epochs[model_name] = {}
    for experiment_number in experiment_numbers:
        print("  experiment_number: ", experiment_number, " of ", experiment_numbers)
        
        # Initialise a subsubdictionary for each module and experiment
        Epochs[model_name][experiment_number] = {}
        
        # Define paths to the cost and accuracy log files
        cost_file_path = "data/new/yue/" + experiment_number + "_cost_" + model_name + ".log"
        acc_file_path = "data/new/yue/" + experiment_number + "_acc_" + model_name + ".log"
        
        cost_file = open(cost_file_path, "r")
        # save the contents of cost_file to variable cost_vectors:
        exec("cost_vectors = [" + cost_file.read() + "]")
        cost_file.close()
        
        # Compute, print, and save means and stds of costs for this experiment
        # and module
        means, stds = return_means_and_stds(cost_vectors)
        print("    The average mean: ", np.mean(means))
        print("    The average std: ", np.mean(stds))
        Means[model_name][experiment_number] = means
        Stds[model_name][experiment_number] = stds
        
        # Compute and save the epoch in which each threshold in 'thresholds' 
        # has been reached
        for threshold in thresholds:
            Epochs[model_name][experiment_number][threshold] = compute_when_threshold_acc_is_reached(acc_file_path, threshold)
        