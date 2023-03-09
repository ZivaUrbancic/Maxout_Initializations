import copy

# Create a RunLog for each run, which is updated with the cost vectors,
# losses and accuracies, and experiment number.

_datasets = ['mnist', 'cifar', 'speechcommands']
_sizes = ['small', 'large']
_ranks = ['relu', 'maxout2', 'maxout3', 'maxout5']

class RunLog:

    def __init__(self, dataset, size, rank,
                 adjust_regions, adjust_variance, experiment_number=0):

        self.dataset = dataset
        self.size = size
        self.rank = rank

        self.experiment_number = experiment_number
        self.adjust_regions = adjust_regions
        self.adjust_variance = adjust_variance
        self.cost_vectors = []
        self.losses = []
        self.accuracies = []

    # Record the experiment number as l.experiment_number = ...

    # Use the following three functions to record the cost_vectors,
    # losses and accuracies at a given epoch and timestep.

    def record_cost_vector(self, epoch, step, cost_vector):
        self.cost_vectors.append([epoch, step, cost_vector])

    def record_losses(self, epoch, step, losses):
        self.losses.append([epoch, step, losses])

    def record_accuracies(self, epoch, step, acc):
        self.accuracies.append([epoch, step, acc])

# The class Log gathers the runlogs together in a big dictionary.
# This is used in each experiment file to collect multiple runs,
# as well as to append Logs together.

class Log:

    def __init__(self, dictionary = None, ranks = _ranks):

        if dictionary is None:
            self.ranks = ranks
            ranks_dict = {a : [] for a in ranks}
            sizes = {s : copy.deepcopy(ranks_dict) for s in _sizes}
            self.dict = {d : copy.deepcopy(sizes) for d in _datasets}
        else:
            self.ranks = [key for key in dictionary['mnist']['small']]
            self.dict = dictionary

    def add_runlog(self, runlog):
        directory = self.dict[runlog.dataset][runlog.size][runlog.rank]
        log_dict = {'experiment_number' : runlog.experiment_number,
                    'adjust_regions' : runlog.adjust_regions,
                    'adjust_variance' : runlog.adjust_variance,
                    'cost_vectors' : runlog.cost_vectors,
                    'losses' : runlog.losses,
                    'accuracies' : runlog.accuracies}
        if log_dict in directory:
            print('experiment number', runlog.experiment_number,
                  'has already been added to the log')
        else:
            directory.append(log_dict)

    def save(self, experiment_number):
        print(self.dict,
                   file=open(str(experiment_number)+".log",'+a'))

    def append(self, log):
        for dataset in _datasets:
            for size in _sizes:
                for rank in _ranks:
                    current = self.dict[dataset][size][rank]
                    updates = log.dict[dataset][size][rank]
                    if len(updates) > 0:
                        print('Appending', len(updates)//3 ,
                              'runs of each initialisation routine for:',
                              dataset, size, rank)
                    for update in updates:
                        # print(update)
                        # if update['experiment_number'] in [c['experiment_number']
                        #                                    for c in current]:
                        #     print('This run of', dataset, size, rank,
                        #               'has already been added to this log')
                        # else:
                        current.append(update)
