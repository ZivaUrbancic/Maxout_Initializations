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

costA = "data/new/yue/858403345_cost_A.log"
costB = "data/new/yue/858403345_cost_B.log"
costC = "data/new/yue/858403345_cost_C.log"
fA = open(costA, "r")
fB = open(costB, "r")
fC = open(costC, "r")

# Uncomment and adjust if cost D should also be analysed:
# costD = "data/new/yue/858403345_cost_D.log"
# fD = open(costD, "r")

exec("cost_vectors_A = [" + fA.read() + "]")
fA.close()

exec("cost_vectors_B = [" + fB.read() + "]")
fB.close()

exec("cost_vectors_C = [" + fC.read() + "]")
fC.close()

# Uncomment and adjust if cost D should also be analysed:
#exec("cost_vectors_D = [" + fD.read() + "]")
#fD.close()

if not len(cost_vectors_A)==len(cost_vectors_B)==len(cost_vectors_C):
    assert False

def return_means_and_stds(cost_vectors):
    means = []
    stds = []
    for i in range(len(cost_vectors)):
        cv = [j for j in cost_vectors[i][1] if j != -1]
        means += [np.mean(cv)]
        stds += [np.std(cv)]
    return means, stds

meanA, stdA = return_means_and_stds(cost_vectors_A)
meanB, stdB = return_means_and_stds(cost_vectors_B)
meanC, stdC = return_means_and_stds(cost_vectors_C)
#meanD, stdD = return_means_and_stds(cost_vectors_D)
    
print("Average mean A: ", np.mean(meanA))
print("Average mean B: ", np.mean(meanB))
print("Average mean C: ", np.mean(meanC))
#print("Average mean D: ", np.mean(meanD))
print("Standard deviation of means of A: ", np.std(meanA))
print("Standard deviation of means of B: ", np.std(meanB))
print("Standard deviation of means of C: ", np.std(meanC))
#print("Standard deviation of means of D: ", np.std(meanD))
print("Average std A: ", np.mean(stdA))
print("Average std B: ", np.mean(stdB))
print("Average std C: ", np.mean(stdC))
#print("Average std D: ", np.mean(stdD))