import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import json

exec(open("../log_classes.py").read())

data_log = Log()

experiment_numbers = [294863938]

for number in experiment_numbers:
    print('Experiment number:', number)
    with open(str(number) + '.log') as file:
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

def epoch_and_step_to_partial_epoch(runs):
    new_list = []
    for run in runs:
        run_list = []
        for epoch, step, value in run:
            partial_epoch = epoch + step/600
            run_list.append([partial_epoch, value])
        new_list.append(np.array(run_list))
    return new_list

def epoch_and_step_to_partial_epoch_accuracies(runs):
    new_list = []
    for run in runs:
        run_list = []
        for epoch, step, value in run:
            partial_epoch = epoch + step/600
            run_list.append([partial_epoch, value[0]])
        new_list.append(np.array(run_list))
    return new_list

losses_std = epoch_and_step_to_partial_epoch(losses_std)
losses_var = epoch_and_step_to_partial_epoch(losses_var)
losses_full = epoch_and_step_to_partial_epoch(losses_full)

accuracies_std = epoch_and_step_to_partial_epoch_accuracies(accuracies_std)
accuracies_var = epoch_and_step_to_partial_epoch_accuracies(accuracies_var)
accuracies_full = epoch_and_step_to_partial_epoch_accuracies(accuracies_full)

#%%

def compute_average_across_runs(losses_vector):

    average_across_runs = []
    n_runs = len(losses_vector)
    n_steps = len(losses_vector[0])
    for step in range(n_steps):
        loss = 0
        for run in range(n_runs):
            loss += losses_vector[run][step][1]
        average_across_runs.append([losses_vector[0][step][0],loss/n_runs])

    return average_across_runs

average_loss_std = compute_average_across_runs(losses_std)
average_loss_var = compute_average_across_runs(losses_var)
average_loss_full = compute_average_across_runs(losses_full)

average_accuracies_std = compute_average_across_runs(accuracies_std)
average_accuracies_var = compute_average_across_runs(accuracies_var)
average_accuracies_full = compute_average_across_runs(accuracies_full)

def compute_average_tcosts_nregions_across_runs(cost_vectors):

    average_tcosts_across_runs = []
    average_nregions_across_runs = []

    n_runs = len(cost_vectors)
    n_steps = len(cost_vectors[0])
    for step in range(n_steps):
        tcost = 0
        nregion = 0
        for run in range(n_runs):
            region_costs = cost_vectors[run][step][2][2]
            for region_cost_entry in region_costs:
                tcost += region_cost_entry[0]*region_cost_entry[2]
                nregion += region_cost_entry[2]

        average_tcosts_across_runs.append(tcost/n_runs)
        average_nregions_across_runs.append(nregion/n_runs)

    return average_tcosts_across_runs, average_nregions_across_runs

average_tcosts_std, average_nregions_std = compute_average_tcosts_nregions_across_runs(cost_vectors_std)
average_tcosts_var, average_nregions_var = compute_average_tcosts_nregions_across_runs(cost_vectors_var)
average_tcosts_full, average_nregions_full = compute_average_tcosts_nregions_across_runs(cost_vectors_full)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

ax1.plot(range(97),average_tcosts_std)
ax1.plot(range(97),average_tcosts_var)
ax1.plot(range(97),average_tcosts_full)
ax1.legend(['std','var','full'])
ax1.set_ylabel('total costs')

ax2.plot(range(97),average_nregions_std)
ax2.plot(range(97),average_nregions_var)
ax2.plot(range(97),average_nregions_full)
ax2.legend(['std','var','full'])
ax2.set_ylabel('number regions')

ax3.plot([i[0] for i in average_loss_std],[i[1] for i in average_loss_std])
ax3.plot([i[0] for i in average_loss_var],[i[1] for i in average_loss_var])
ax3.plot([i[0] for i in average_loss_full],[i[1] for i in average_loss_full])
ax3.legend(['std','var','full'])
ax3.set_ylabel('loss')

ax4.plot([i[0] for i in average_accuracies_std],[i[1] for i in average_accuracies_std])
ax4.plot([i[0] for i in average_accuracies_var],[i[1] for i in average_accuracies_var])
ax4.plot([i[0] for i in average_accuracies_full],[i[1] for i in average_accuracies_full])
ax4.legend(['std','var','full'])
ax4.set_ylabel('accuracy')

plt.show()
