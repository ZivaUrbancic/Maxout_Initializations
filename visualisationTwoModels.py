import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import json

exec(open("log_classes.py").read())

data_log = Log()

experiment_numbers = ["528354401","673594811"]

epoch_start = 0
epoch_end = 24 # can be float


for number in experiment_numbers:
    print('Experiment number:', number)
    with open('experiments/' + str(number) + '.log') as file:
        file_string = file.read()
        file_string = file_string.replace("'", "\"")
        file_string = file_string.replace("True", "true")
        file_string = file_string.replace("False", "false")
    file_log = Log(json.loads(file_string))
    data_log.append(file_log)

init_std = []
init_full = []

for training in data_log.dict['mnist']['small']['relu']:
    if not training['adjust_variance']:
        init_std.append(training)
    elif training['adjust_regions']:
        init_full.append(training)

cost_vectors_std = [run['cost_vectors'] for run in init_std]
cost_vectors_full = [run['cost_vectors'] for run in init_full]

losses_std = [run['losses'] for run in init_std]
losses_full = [run['losses'] for run in init_full]

accuracies_std = [run['accuracies'] for run in init_std]
accuracies_full = [run['accuracies'] for run in init_full]

def n_logpoints_per_epoch(runs):
    single_run = runs[0]
    n_epochs = single_run[-1][0]+1
    n_logpoints_post_initialisation = len(single_run)-1
    return n_logpoints_post_initialisation/n_epochs

def drop_all_accuracies_except_total(runs):
    new_list = []
    for run in runs:
        run_list = []
        for epoch, step, value in run:
            run_list.append([epoch,step,[value[0][0],value[1][0]]]) # 0 entry for total accuracy
        new_list.append(np.array(run_list))
    return new_list

def epoch_and_step_to_partial_epoch(runs,n_logs_per_epoch=-1):
    if n_logs_per_epoch<0:
        n_logs_per_epoch = n_logpoints_per_epoch(runs)
    new_list = []
    for run in runs:
        run_list = []
        for i,(epoch, step, value) in enumerate(run):
            partial_epoch = i / n_logs_per_epoch
            run_list.append([partial_epoch, value])
        new_list.append(np.array(run_list))
    return new_list

losses_std = epoch_and_step_to_partial_epoch(losses_std)
losses_full = epoch_and_step_to_partial_epoch(losses_full)

accuracy_std = drop_all_accuracies_except_total(accuracies_std)
accuracy_full = drop_all_accuracies_except_total(accuracies_full)
accuracy_std = epoch_and_step_to_partial_epoch(accuracy_std)
accuracy_full = epoch_and_step_to_partial_epoch(accuracy_full)

#%%

def compute_average_across_runs(cost_vector):
    average_across_runs_train = []
    average_across_runs_test = []
    n_runs = len(cost_vector)
    n_steps = len(cost_vector[0])
    for step in range(n_steps):
        loss_train = 0
        loss_test = 0
        for run in range(n_runs):
            loss_train += cost_vector[run][step][1][0]
            loss_test += cost_vector[run][step][1][1]
        average_across_runs_train.append([cost_vector[0][step][0],loss_train/n_runs])
        average_across_runs_test.append([cost_vector[0][step][0],loss_test/n_runs])
    return np.array(average_across_runs_train),np.array(average_across_runs_test)

average_loss_train_std, average_loss_test_std = compute_average_across_runs(losses_std)
average_loss_train_full, average_loss_test_full = compute_average_across_runs(losses_full)

average_accuracy_train_std, average_accuracy_test_std = compute_average_across_runs(accuracy_std)
average_accuracy_train_full, average_accuracy_test_full = compute_average_across_runs(accuracy_full)

def compute_average_tcosts_nregions_across_runs(cost_vectors):

    average_tcosts_train = []
    average_nregions_train = []
    average_tcosts_test = []
    average_nregions_test = []

    n_runs = len(cost_vectors)
    n_steps = len(cost_vectors[0])
    for step in range(n_steps):
        tcost_train = 0
        nregion_train = 0
        tcost_test = 0
        nregion_test = 0
        for run in range(n_runs):
            region_costs_train = cost_vectors[run][step][2][0][-1] # final entry i = up to layer i
            region_costs_test = cost_vectors[run][step][2][1][-1] # final entry i = up to layer i
            for region_cost_entry in region_costs_train:
                tcost_train += region_cost_entry[0]*region_cost_entry[2]
                nregion_train += region_cost_entry[2]
            for region_cost_entry in region_costs_test:
                tcost_test += region_cost_entry[0]*region_cost_entry[2]
                nregion_test += region_cost_entry[2]

        average_tcosts_train.append(tcost_train/n_runs)
        average_nregions_train.append(nregion_train/n_runs)
        average_tcosts_test.append(tcost_test/n_runs)
        average_nregions_test.append(nregion_test/n_runs)

    return np.array(average_tcosts_train), np.array(average_nregions_train), np.array(average_tcosts_test), np.array(average_nregions_test)

def cum_cardinality(cost_vector):

    # change third column from number of regions
    # to number of points in regions
    cost_array = np.array(cost_vector)
    # remove column 0 containing cost
    cardinality_array = cost_array[:,1:]
    # sort rows by first entry (=cardinality) in ascending order
    cardinality_array = cardinality_array[np.argsort(cardinality_array[:,0])]
    # compute cummultative cardinalities
    cardinality_array[:,1] = np.cumsum(cardinality_array[:,1])
    # find unique cardinalities
    _,indices = np.unique(cardinality_array[::-1,0],return_index=True)
    cardinality_array = cardinality_array[-1-indices]
    return cardinality_array


def cum_cost(cost_vector):

    # change third column from number of regions
    # to number of points in regions
    cost_array = np.array(cost_vector)
    # remove column 1 containing cardinality
    cost_array = cost_array[:,[0,2]]
    # sort rows by first entry (=cost) in ascending order
    cost_array = cost_array[np.argsort(cost_array[:,0])]
    # compute cummultative cost
    cost_array[:,1] = np.cumsum(cost_array[:,1])
    # find unique cost
    _,indices = np.unique(cost_array[::-1,0],return_index=True)
    cost_array = cost_array[-1-indices]

    # prepend (0,0) if first entry has non-zero cost
    if cost_array[0][0] > 0:
        cost_array = np.concatenate((np.array([[0,0]]),cost_array))

    return cost_array

# def n_regions_in_hyperplane_arrangement(ambient_dim,n_hyperplanes):
#     return np.sum([math.comb(n_hyperplanes,i) for i in range(ambient_dim+1)])

def pareto_card(cost_vector):
    cost_array = np.array(cost_vector)
    n_regions = np.sum(cost_array[:,2])
    mincard = np.min(cost_array[:,2])
    return n_regions/np.sum(np.log(cost_array[:,1]/mincard)*cost_array[:,2])
    # return n_regions/np.sum(cost_array[:,1]*cost_array[:,2]) # exponential distribution

def compute_pareto_across_runs(cost_vector):
    pareto_across_runs_train = []
    pareto_across_runs_test = []
    n_runs = len(cost_vector)
    n_steps = len(cost_vector[0])
    for step in range(n_steps):
        pareto_train = 0
        pareto_test = 0
        for run in range(n_runs):
            vector_train = cost_vector[run][step][1][0][-1]
            vector_test = cost_vector[run][step][1][1][-1]
            pareto_train += pareto_card(vector_train)
            pareto_test += pareto_card(vector_test)
        pareto_across_runs_train.append([cost_vector[0][step][0],pareto_train/n_runs])
        pareto_across_runs_test.append([cost_vector[0][step][0],pareto_test/n_runs])
    return np.array(pareto_across_runs_train),np.array(pareto_across_runs_test)

costs_std = epoch_and_step_to_partial_epoch(cost_vectors_std)
costs_full = epoch_and_step_to_partial_epoch(cost_vectors_full)



# cost_vector = cost_vectors_std[0][0][2][0][-1]

# # cdfCost = cdf_cost(cost_vector)
# # plt.plot(cdfCost[:,0],cdfCost[:,1])
# # plt.show()

# cdfCard = cum_cardinality(cost_vector)
# plt.plot(cdfCard[1:,0],cdfCard[1:,1])
# plt.show()

# cost_vector = cost_vectors_full[0][0][2][0][-1]
# cdfCard = cum_cardinality(cost_vector)
# plt.plot(cdfCard[1:,0],cdfCard[1:,1])
# plt.show()

# # ML estimator for pareto parameter alpha:
# cdfCard[-1,1]/np.sum(np.log(cdfCard[:,0]/cdfCard[0,0])*cdfCard[:,1])

#-------------------------------------- PLOTTING CODE ---------------------------------------

average_tcosts_std_train, average_nregions_std_train, average_tcosts_std_test, average_nregions_std_test = compute_average_tcosts_nregions_across_runs(cost_vectors_std)
average_tcosts_full_train, average_nregions_full_train, average_tcosts_full_test, average_nregions_full_test = compute_average_tcosts_nregions_across_runs(cost_vectors_full)

pareto_train_std, pareto_test_std = compute_pareto_across_runs(costs_std)
pareto_train_full, pareto_test_full = compute_pareto_across_runs(costs_full)

fig, subfigures = plt.subplots(4,2, sharex=True, figsize =(20,20))#,sharey='row')

subfigures[0,0].plot(np.linspace(epoch_start,epoch_end,len(average_tcosts_std_train)),average_tcosts_std_train)
subfigures[0,0].plot(np.linspace(epoch_start,epoch_end,len(average_tcosts_full_train)),average_tcosts_full_train)
subfigures[0,0].legend(['std','full'])
subfigures[0,0].set_ylabel('total costs (train)')
subfigures[0,1].plot(np.linspace(epoch_start,epoch_end,len(average_tcosts_std_test)),average_tcosts_std_test)
subfigures[0,1].plot(np.linspace(epoch_start,epoch_end,len(average_tcosts_full_test)),average_tcosts_full_test)
subfigures[0,1].legend(['std','full'])
subfigures[0,1].set_ylabel('total costs (test)')

subfigures[1,0].plot(np.linspace(epoch_start,epoch_end,len(average_nregions_std_train)),average_nregions_std_train)
subfigures[1,0].plot(np.linspace(epoch_start,epoch_end,len(average_nregions_full_train)),average_nregions_full_train)
subfigures[1,0].legend(['std','full'])
subfigures[1,0].set_ylabel('number of regions (train)')
subfigures[1,1].plot(np.linspace(epoch_start,epoch_end,len(average_nregions_std_test)),average_nregions_std_test)
subfigures[1,1].plot(np.linspace(epoch_start,epoch_end,len(average_nregions_full_test)),average_nregions_full_test)
subfigures[1,1].legend(['std','full'])
subfigures[1,1].set_ylabel('number of regions (test)')

subfigures[2,0].plot(np.linspace(epoch_start,epoch_end,len(average_accuracy_train_std)),[x[1] for x in average_accuracy_train_std])
subfigures[2,0].plot(np.linspace(epoch_start,epoch_end,len(average_accuracy_train_full)),[x[1] for x in average_accuracy_train_full])
subfigures[2,0].legend(['std','full'])
subfigures[2,0].set_ylabel('accuracy (train)')
subfigures[2,1].plot(np.linspace(epoch_start,epoch_end,len(average_accuracy_test_std)),[x[1] for x in average_accuracy_test_std])
subfigures[2,1].plot(np.linspace(epoch_start,epoch_end,len(average_accuracy_test_full)),[x[1] for x in average_accuracy_test_full])
subfigures[2,1].legend(['std','full'])
subfigures[2,1].set_ylabel('accuracy (test)')
subfigures[2,0].set_ylim([0.8,1])
subfigures[2,1].set_ylim([0.8,1])

subfigures[3,0].plot(np.linspace(epoch_start,epoch_end,len(average_loss_train_std)),[x[1] for x in average_loss_train_std])
subfigures[3,0].plot(np.linspace(epoch_start,epoch_end,len(average_loss_train_full)),[x[1] for x in average_loss_train_full])
subfigures[3,0].legend(['std','full'])
subfigures[3,0].set_ylabel('loss (train)')
subfigures[3,1].plot(np.linspace(epoch_start,epoch_end,len(average_loss_test_std)),[x[1] for x in average_loss_test_std])
subfigures[3,1].plot(np.linspace(epoch_start,epoch_end,len(average_loss_test_full)),[x[1] for x in average_loss_test_full])
subfigures[3,1].legend(['std','var','full'])
subfigures[3,1].set_ylabel('loss (test)')
subfigures[3,0].set_ylim([0,1.5])
subfigures[3,1].set_ylim([0,1.5])

# Pareto estimator:
# subfigures[2,0].plot(np.linspace(epoch_start,epoch_end,len(pareto_train_std)),[x[1] for x in pareto_train_std])
# subfigures[2,0].plot(np.linspace(epoch_start,epoch_end,len(pareto_train_var)),[x[1] for x in pareto_train_var])
# subfigures[2,0].plot(np.linspace(epoch_start,epoch_end,len(pareto_train_full)),[x[1] for x in pareto_train_full])
# subfigures[2,0].legend(['std','var','full'])
# subfigures[2,0].set_ylabel('pareto (train)')
# subfigures[2,1].plot(np.linspace(epoch_start,epoch_end,len(pareto_test_std)),[x[1] for x in pareto_test_std])
# subfigures[2,1].plot(np.linspace(epoch_start,epoch_end,len(pareto_test_var)),[x[1] for x in pareto_test_var])
# subfigures[2,1].plot(np.linspace(epoch_start,epoch_end,len(pareto_test_full)),[x[1] for x in pareto_test_full])
# subfigures[2,1].legend(['std','var','full'])
# subfigures[2,1].set_ylabel('pareto (test)')


plt.show()
# ---------------------------------------------------------------------------------------------------

# S_std = [np.sum([a[2] for a in cost_vector[-1][2][0][-1]]) for cost_vector in cost_vectors_std]
# S_var = [np.sum([a[2] for a in cost_vector[-1][2][0][-1]]) for cost_vector in cost_vectors_var]
# S_full = [np.sum([a[2] for a in cost_vector[-1][2][0][-1]]) for cost_vector in cost_vectors_full]

# print('means: std, var, full')
# print(np.mean(S_std),np.mean(S_var),np.mean(S_full))
# print('standard_deviation: std, var, full')
# print(np.std(S_std),np.std(S_var),np.std(S_full))

# print("std: ",average_accuracy_train_std[-1][1]-average_accuracy_test_std[-1][1])
# print("var: ",average_accuracy_train_var[-1][1]-average_accuracy_test_var[-1][1])
# print("full: ",average_accuracy_train_full[-1][1] - average_accuracy_test_full[-1][1])


# plotting cardinality

# def cdf_cardinality(cost_vector):

#     cardinality_array = cum_cardinality(cost_vector)
#     # normalisation
#     cardinality_array[:,1] /= cardinality_array[-1,1]
#     return cardinality_array

# def pdf_cardinality(cost_vector):
#     cdf = cdf_cardinality(cost_vector)
#     pdf = cdf
#     pdf[1:,1] = cdf[1:,1]-cdf[:-1,1]
#     return pdf

# # cost_vector_a = cost_vectors_std[0][0][2][0][-1]
# # cost_vector_b = cost_vectors_std[0][40][2][0][-1]
# # cost_vector_c = cost_vectors_std[0][-1][2][0][-1]
# # pdf_a = pdf_cardinality(cost_vector_a)
# # pdf_b = pdf_cardinality(cost_vector_b)
# # pdf_c = pdf_cardinality(cost_vector_c)
# # plt.plot(pdf_a[:12,0],pdf_a[:12,1])
# # plt.plot(pdf_b[:12,0],pdf_b[:12,1])
# # plt.plot(pdf_c[:12,0],pdf_c[:12,1])
# # plt.show()

# def exp_lambda(cost_vector):
#     cost_array = np.array(cost_vector)
#     return np.sum(cost_array[:,2])/np.sum(cost_array[:,1]*cost_array[:,2]) # exponential distribution

# def exp_df(X,lamb):
#     return [lamb*np.exp(-lamb*x) for x in X]


# cost_vector = cost_vectors_std[0][-1][2][0][-1]
# pdf = pdf_cardinality(cost_vector)
# plt.plot(pdf[:12,0],pdf[:12,1])
# lamb = exp_lambda(cost_vector)
# plt.plot(pdf[:12,0],exp_df(pdf[:12,0],lamb))
# plt.show()
