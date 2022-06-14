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
    f1 = experiment_number+"_"+acc_or_loss+"_A.log"
    f2 = experiment_number+"_"+acc_or_loss+"_B.log"
    D = read_data_from_exp(f1, num_runs, num_epochs)
    R = read_data_from_exp(f2, num_runs, num_epochs)
    return D, R

def compute_Y1_Y2_Y3(experiment_number, acc_or_loss, num_runs, num_epochs):
    f1 = experiment_number+"_"+acc_or_loss+"_A.log"
    f2 = experiment_number+"_"+acc_or_loss+"_B.log"
    f3 = experiment_number+"_"+acc_or_loss+"_C.log"
    Y1 = read_data_from_exp(f1, num_runs, num_epochs)
    Y2 = read_data_from_exp(f2, num_runs, num_epochs)
    Y3 = read_data_from_exp(f3, num_runs, num_epochs)
    return Y1, Y2, Y3

def compute_Ys(experiment_number, models, acc_or_loss, num_runs, num_epochs):
    fs = [experiment_number+"_"+acc_or_loss+"_"+model+".log" for model in models]
    Ys = [read_data_from_exp(fi, num_runs, num_epochs) for fi in fs]
    return Ys

def visualiseOld2(Y1, Y2, Y1_std=None, Y2_std=None):
    X = np.linspace(0, num_epochs, num=len(Y1))
    plt.plot(X, Y1, "royalblue")
    plt.plot(X, Y2, "orange")
    if not (Y1_std is None or Y2_std is None):
        plt.fill_between(X, Y1-Y1_std, Y1+Y1_std, color="skyblue")
        plt.fill_between(X, Y2-Y2_std, Y2+Y2_std, color="navajowhite")

def visualiseOld3(Y1, Y2, Y3, Y1_std=None, Y2_std=None, Y3_std=None):
    X = np.linspace(0, num_epochs, num=len(Y1))
    plt.plot(X, Y1, "r")
    plt.plot(X, Y2, "b")
    plt.plot(X, Y3, "g")
    if not (Y1_std is None or Y2_std is None or Y3_std is None):
        plt.fill_between(X, Y1-Y1_std, Y1+Y1_std, color="skyblue")
        plt.fill_between(X, Y2-Y2_std, Y2+Y2_std, color="navajowhite")
        plt.fill_between(X, Y3-Y3_std, Y3+Y3_std, color="navajowhite")

def visualise(Ys, Ys_std=None):
    X = np.linspace(0, num_epochs, num=len(Ys[1]))
    colours = ["red","blue","green","darkorange"]
    colours_std = ["mistyrose","lightskyblue","palegreen","bisque"]

    for Yi,colour in zip(Ys,colours):
        plt.plot(X, Yi, colour)

    if Ys_std != None:
        for Yi,Yi_std,colour_std in zip(Ys,Ys_std,colours_std):
            plt.fill_between(X, Yi+Yi_std, Yi-Yi_std, color=colour_std)

#experiment_number = "491331889", "248360644"
e = ["989642057"]
models = ["A", "B", "C", "D"]
Ys = [[] for model in models]
Zs = [[] for model in models]
for experiment_number in e:

    f = open(experiment_number+".log", "r")
    for i, l in enumerate(f):
        if i == 6:
            _, s = l.split(": ")
            num_runs = int(s.strip())
        elif i == 7:
            s = l.split(" ")
            num_epochs = int(s[2])

    ys = compute_Ys(experiment_number, models, "acc", num_runs, num_epochs)
    zs = compute_Ys(experiment_number, models, "loss", num_runs, num_epochs)
    if len(Ys[0])==0:
        Ys = [np.array(yi) for yi in ys]
        Zs = [np.array(zi) for zi in zs]
    else:
        for i in range(len(Ys)):
            Ys[i] = np.concatenate((Ys[i], np.array(ys[i])), axis = 0)
            Zs[i] = np.concatenate((Zs[i], np.array(zs[i])), axis = 0)

Ys_std = [np.array(Yi).std(axis=0) for Yi in Ys]
Zs_std = [np.array(Zi).std(axis=0) for Zi in Zs]
Ys = [Yi.mean(axis=0) for Yi in Ys]
Zs = [Zi.mean(axis=0) for Zi in Zs]

plt.figure(0)
visualise(Ys, Ys_std)
plt.figure(1)
visualise(Zs, Zs_std)
plt.show()
