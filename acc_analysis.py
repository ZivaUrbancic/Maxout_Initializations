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

alpha = 90 #threshold
accA = "data/new/yue/858403345_acc_A.log"
accB = "data/new/yue/858403345_acc_B.log"
accC = "data/new/yue/858403345_acc_C.log"
#accD = "data/new/yue/858403345_acc_D.log"

def compute_when_threshold_acc_is_reached(file, threshold):
    #X = np.zeros((num_runs, 50*num_epochs))
    f = open(file, "r")
    #run_shift = 0
    #n_run = 0
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
        #print(b)
        #assert False
        if b >= threshold:
            skip = True
            #print("Found one")
            n_steps_to_threshold += [float(i-start_id)]
        if int(a[0]) > n_run:
            if int(a[0]) > len(n_steps_to_threshold):
                n_steps_to_threshold += [float(i-start_id)]    
            skip = False
            start_id = i
            n_run = int(a[0])
#    for i, l in enumerate(f):
#        print("In second for")
#        a, b = l.split("], [")
#        a = a[2:].split(", ") # a = [num_run, num_epoch, num_step]
#        print(type(a[1]))
#        assert False
#        if int(a[1]) == 1:
#            num_steps_per_epoch = i
#            break
            #print(num_steps_per_epoch)
    return 1/num_steps_per_epoch * np.array(n_steps_to_threshold)


A = compute_when_threshold_acc_is_reached(accA, alpha)
B = compute_when_threshold_acc_is_reached(accB, alpha)
C = compute_when_threshold_acc_is_reached(accC, alpha)
#D = compute_when_threshold_acc_is_reached(accD, alpha)

print("Threshold ", alpha, " reached after: ")
print("  ", np.mean(A), " epochs for A with std ", np.std(A))
print("  ", np.mean(B), " epochs for B with std ", np.std(B))
print("  ", np.mean(C), " epochs for C with std ", np.std(C))
#print("     ", np.mean(D), " epochs for D with std ", np.std(D))
