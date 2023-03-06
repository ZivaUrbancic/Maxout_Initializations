import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import json

exec(open("log_classes.py").read())

data_log = Log()

experiment_numbers = ["342911688"]

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
init_var = []
init_full = []

for training in data_log.dict['mnist']['small']['relu']:
    if not training['adjust_variance']:
        init_std.append(training)
    elif training['adjust_regions']:
        init_full.append(training)
    else:
        init_var.append(training)

cost_vectors_std = [run['cost_vectors'] for run in init_std]
cost_vectors_var = [run['cost_vectors'] for run in init_var]
cost_vectors_full = [run['cost_vectors'] for run in init_full]

losses_std = [run['losses'] for run in init_std]
losses_var = [run['losses'] for run in init_var]
losses_full = [run['losses'] for run in init_full]

accuracies_std = [run['accuracies'] for run in init_std]
accuracies_var = [run['accuracies'] for run in init_var]
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
losses_var = epoch_and_step_to_partial_epoch(losses_var)
losses_full = epoch_and_step_to_partial_epoch(losses_full)

accuracies_std = drop_all_accuracies_except_total(accuracies_std)
accuracies_var = drop_all_accuracies_except_total(accuracies_var)
accuracies_full = drop_all_accuracies_except_total(accuracies_full)
accuracies_std = epoch_and_step_to_partial_epoch(accuracies_std)
accuracies_var = epoch_and_step_to_partial_epoch(accuracies_var)
accuracies_full = epoch_and_step_to_partial_epoch(accuracies_full)

#%%

def compute_average_across_runs(losses_vector):
    average_across_runs_train = []
    average_across_runs_test = []
    n_runs = len(losses_vector)
    n_steps = len(losses_vector[0])
    for step in range(n_steps):
        loss_train = 0
        loss_test = 0
        for run in range(n_runs):
            loss_train += losses_vector[run][step][1][0]
            loss_test += losses_vector[run][step][1][1]
        average_across_runs_train.append([losses_vector[0][step][0],loss_train/n_runs])
        average_across_runs_test.append([losses_vector[0][step][0],loss_test/n_runs])
    return average_across_runs_train,average_across_runs_test

average_loss_train_std, average_loss_test_std = compute_average_across_runs(losses_std)
average_loss_var = compute_average_across_runs(losses_var)
average_loss_full = compute_average_across_runs(losses_full)

average_accuracies_train_std, average_accuracies_test_std = compute_average_across_runs(accuracies_std)
average_accuracies_train_var, average_accuracies_test_var = compute_average_across_runs(accuracies_var)
average_accuracies_train_full, average_accuracies_test_full = compute_average_across_runs(accuracies_full)

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
            region_costs = cost_vectors[run][step][2][0] # final entry i = up to layer i
            for region_cost_entry in region_costs:
                tcost += region_cost_entry[0]*region_cost_entry[2]
                nregion += region_cost_entry[2]

        average_tcosts_across_runs.append(tcost/n_runs)
        average_nregions_across_runs.append(nregion/n_runs)

    return average_tcosts_across_runs, average_nregions_across_runs

average_tcosts_std, average_nregions_std = compute_average_tcosts_nregions_across_runs(cost_vectors_std)
average_tcosts_var, average_nregions_var = compute_average_tcosts_nregions_across_runs(cost_vectors_var)
average_tcosts_full, average_nregions_full = compute_average_tcosts_nregions_across_runs(cost_vectors_full)

# # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

# # ax1.plot(range(97),average_tcosts_std)
# # ax1.plot(range(97),average_tcosts_var)
# # ax1.plot(range(97),average_tcosts_full)
# # ax1.legend(['std','var','full'])
# # ax1.set_ylabel('total costs')

# # ax2.plot(range(97),average_nregions_std)
# # ax2.plot(range(97),average_nregions_var)
# # ax2.plot(range(97),average_nregions_full)
# # ax2.legend(['std','var','full'])
# # ax2.set_ylabel('number regions')

# # ax3.plot([i[0] for i in average_loss_std],[i[1] for i in average_loss_std])
# # ax3.plot([i[0] for i in average_loss_var],[i[1] for i in average_loss_var])
# # ax3.plot([i[0] for i in average_loss_full],[i[1] for i in average_loss_full])
# # ax3.legend(['std','var','full'])
# # ax3.set_ylabel('loss')

# # ax4.plot([i[0] for i in average_accuracies_std],[i[1] for i in average_accuracies_std])
# # ax4.plot([i[0] for i in average_accuracies_var],[i[1] for i in average_accuracies_var])
# # ax4.plot([i[0] for i in average_accuracies_full],[i[1] for i in average_accuracies_full])
# # ax4.legend(['std','var','full'])
# # ax4.set_ylabel('accuracy')

# # plt.show()
