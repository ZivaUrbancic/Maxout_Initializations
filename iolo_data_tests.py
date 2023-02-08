import numpy as np
import scipy as sp

import json

exec(open("log_classes.py").read())

data_log = Log()

experiment_numbers = [614482321]

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
        

