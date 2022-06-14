import copy

# Create a RunLog for each run, which is updated with the cost vector,
# loss and accuracy, and experiment number.

# The class Log gathers the runlogs together in a big dictionary.
# This is used in each experiment file to collect multiple runs,
# as well as to append Logs together.

class RunLog:
    
    def __init__(self, dataset, size, rank, 
                 adjust_regions, adjust_variance):
        self.dataset = dataset
        self.size = size
        self.rank = rank
        self.adjust_regions = adjust_regions
        self.adjust_variance = adjust_variance
        self.cost_vector = None
        self.experiment_number = 0
        self.loss = []
        self.accuracy = []
        
    # If the cost vector is being saved to the log L then
    # write l.cost_vector = ... else leave as None
    
    # Record the experiment number as l.experiement_number = ...
    
    # Use the following two functions to record the loss
    # and accuracy at a given epoch and timestep.
        
    def record_loss(self, epoch, step, loss):
        self.loss.append([epoch, step, loss])
        
    def record_accuracy(self, epoch, step, acc):
        self.loss.append([epoch, step, acc])
        
class Log:
    
    def __init__(self, ranks = ['relu',
                                      'maxout2',
                                      'maxout3',
                                      'maxout5']):
        self.ranks = ranks
        ranks_dict = {a : [] for a in ranks}
        sizes = {s : copy.deepcopy(ranks_dict) for s in ['small',
                                            'large']}
        self.dict = {d : copy.deepcopy(sizes) for d in ['mnist',
                                                'cifar',
                                                'speechcommands']}
        
    def add_runlog(self, runlog):
        directory = self.dict[runlog.dataset][runlog.size][runlog.rank]
        log_dict = {'experiment_number' : runlog.experiment_number,
                    'adjust_regions' : runlog.adjust_regions,
                    'adjust_variance' : runlog.adjust_variance,
                    'cost_vector' : runlog.cost_vector,
                    'loss' : runlog.loss,
                    'accuracy' : runlog.accuracy}
        if log_dict in directory:
            print('experiment number', runlog.experiment_number,
                  'has already been added to the log')
        else:
            directory.append(log_dict)
        
    def save(self, experiment_number):
        print(self.dict,
                   file=open(str(experiment_number)+".log",'+a'))
            
    def append(self, log):
        for dataset in ['mnist', 'cifar', 'speechcommands']:
            for size in ['small', 'large']:
                for rank in self.ranks:
                    current = self.dict[dataset][size][rank]
                    update = log.dict[dataset][size][rank]
                    if update.experiement_number in [c.experiement_number
                                                     for c in current]:
                        print('This run of', dataset, size, rank,
                              'has already been added to this log')
                    else:
                        current.append(update)
        
        
        
        
        
        
        
        
        
        
        
        
        
        