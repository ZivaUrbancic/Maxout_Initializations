#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 08:44:14 2022

@author: ziva
"""

import numpy as np
import matplotlib.pyplot as plt

def read_data_from_exp(file, num_runs, num_epochs):
    X = np.zeros((num_runs, 60*num_epochs))
    f = open(file, "r")
    for i, l in enumerate(f):
        a, b = l.split("], [")
        a = a[2:].split(", ")
        b = b[:-3].split(", ")
        n_run = int(a[0])
        n_epoch = int(a[1])
        n_entry = (int(a[2])+1)/10 + n_epoch*60
        n_entry = int(n_entry - 1)
        X[n_run, n_entry] = float(b[0])
    return X

def compute_Y1_Y2(experiment_number, acc_or_loss, num_runs, num_epochs):
    f1 = experiment_number+"_"+acc_or_loss+"_default.log"
    f2 = experiment_number+"_"+acc_or_loss+"_reinit.log"
    D = read_data_from_exp(f1, num_runs, num_epochs)
    R = read_data_from_exp(f2, num_runs, num_epochs)
    return D, R

def compute_Y1_Y2_Y3(experiment_number, acc_or_loss, num_runs, num_epochs):
    f1 = experiment_number+"_"+acc_or_loss+"_default.log"
    f2 = experiment_number+"_"+acc_or_loss+"_rescale.log"
    f3 = experiment_number+"_"+acc_or_loss+"_reinit.log"
    Y1 = read_data_from_exp(f1, num_runs, num_epochs)
    Y2 = read_data_from_exp(f2, num_runs, num_epochs)
    Y3 = read_data_from_exp(f3, num_runs, num_epochs)
    return Y1, Y2, Y3

def visualiseOld(Y1, Y2, Y1_std=None, Y2_std=None):
    X = np.linspace(0, num_epochs, num=len(Y1))
    plt.plot(X, Y1, "royalblue")
    plt.plot(X, Y2, "orange")
    if not (Y1_std is None or Y2_std is None):
        plt.fill_between(X, Y1-Y1_std, Y1+Y1_std, color="skyblue")
        plt.fill_between(X, Y2-Y2_std, Y2+Y2_std, color="navajowhite")

def visualise(Y1, Y2, Y3, Y1_std=None, Y2_std=None, Y3_std=None):
    X = np.linspace(0, num_epochs, num=len(Y1))
    plt.plot(X, Y1, "r")
    plt.plot(X, Y2, "b")
    plt.plot(X, Y3, "g")
    if not (Y1_std is None or Y2_std is None or Y3_std is None):
        plt.fill_between(X, Y1-Y1_std, Y1+Y1_std, color="skyblue")
        plt.fill_between(X, Y2-Y2_std, Y2+Y2_std, color="navajowhite")
        plt.fill_between(X, Y3-Y3_std, Y3+Y3_std, color="navajowhite")

#experiment_number = "491331889", "248360644"
e = ["614754698"]
Y1 = []
Y2 = []
Y3 = []
Z1 = []
Z2 = []
Z3 = []
for experiment_number in e:

    f = open(experiment_number+".log", "r")
    for i, l in enumerate(f):
        if i == 6:
            _, s = l.split(": ")
            num_runs = int(s.strip())
        elif i == 7:
            s = l.split(" ")
            num_epochs = int(s[2])

    y1, y2, y3 = compute_Y1_Y2_Y3(experiment_number, "acc", num_runs, num_epochs)
    z1, z2, z3 =  compute_Y1_Y2_Y3(experiment_number, "loss", num_runs, num_epochs)
    if len(Y1)==0:
        Y1 = np.array(y1)
        Y2 = np.array(y2)
        Y3 = np.array(y3)
        Z1 = np.array(z1)
        Z2 = np.array(z2)
        Z3 = np.array(z3)
    else:
        Y1 = np.concatenate((Y1, np.array(y1)), axis = 0)
        Y2 = np.concatenate((Y2, np.array(y2)), axis = 0)
        Y3 = np.concatenate((Y3, np.array(y3)), axis = 0)
        Z1 = np.concatenate((Z1, np.array(z1)), axis = 0)
        Z2 = np.concatenate((Z2, np.array(z2)), axis = 0)
        Z3 = np.concatenate((Z3, np.array(z3)), axis = 0)

Y1_std = Y1.std(axis=0)
Y2_std = Y2.std(axis=0)
Y3_std = Y3.std(axis=0)
Z1_std = Z1.std(axis=0)
Z2_std = Z2.std(axis=0)
Z3_std = Z3.std(axis=0)
Y1 = Y1.mean(axis=0)
Y2 = Y2.mean(axis=0)
Y3 = Y3.mean(axis=0)
Z1 = Z1.mean(axis=0)
Z2 = Z2.mean(axis=0)
Z3 = Z3.mean(axis=0)

plt.figure(0)
visualise(Y1, Y2, Y3)#, Y1_std, Y2_std, Y3_std)
plt.figure(1)
visualise(Z1, Z2, Z3)#, Z1_std, Z2_std, Z3_std)
plt.show()

print(Y1[-1], Y2[-1], Y3[-1])
print(Z1[-1], Z2[-1], Z3[-1])
